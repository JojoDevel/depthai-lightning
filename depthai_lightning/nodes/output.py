""" High-level output nodes """
from __future__ import annotations

from enum import Enum
from pathlib import Path

import cv2
import depthai as dai

from depthai_lightning.depthai_lightning import PipelineManager

from .base import InputOutput, Node
from .input import Camera


class Codec(Enum):
    """Codecs supported by OAK-D"""

    MJPEG = (1,)
    H264 = (2,)
    H265 = 3


class EncodingConfig(Enum):
    """Configure encoding quality"""

    def __init__(self, codec: Codec, quality: int = None, bitrate: int = None):
        """Create an encoding config

        Args:
            codec (Codec): encoding codec.
            quality (int, optional): quality [0...100(full quality)] when MJPEG is used. Defaults to None (use OAK-D default).
            bitrate (int, optional): bitrate in Kbps for h26x encoding. Defaults to None (use OAK-D default).
        """
        self.codec = codec
        self.quality = quality
        self.bitrate = bitrate

    def apply(self, dai_encoder: dai.node.VideoEncoder, fps: int):
        """Apply the configuration to the depthai encoder object

        Args:
            dai_encoder (dai.node.VideoEncoder): the depthai encoder object, e.g. created by pipeline.createVideoEncoder()
            fps (int): the framerate used for encoding
        """

        # lookup dai internal profiles
        profile_lookup = {
            Codec.H264: dai.VideoEncoderProperties.Profile.H264_MAIN,
            Codec.H265: dai.VideoEncoderProperties.Profile.H265_MAIN,
            Codec.MJPEG: dai.VideoEncoderProperties.Profile.MJPEG,
        }

        # apply the profile to encoder object
        profile = profile_lookup[self.codec]
        dai_encoder.setDefaultProfilePreset(fps, profile)

        # set quality or bitrate
        if self.quality == 100:
            dai_encoder.setLossless(True)
        elif self.quality:
            dai_encoder.setQuality(self.quality)
        elif self.bitrate:
            dai_encoder.setBitrateKbps(self.bitrate)

    def __repr__(self) -> str:
        return f"codec: {self.codec}, quality: {self.quality}, bitrate: {self.bitrate}"

    # some default configs
    BEST = (Codec.MJPEG, 100, None)  # Lossless MJPEG
    HIGH = (Codec.MJPEG, 97, None)  # MJPEG Quality=97 (default)
    MEDIUM = (Codec.MJPEG, 97, None)  # MJPEG Quality=93
    LOW = (Codec.H265, None, 10000)  # H265 BitrateKbps=10000
    HIGH_VIDEO = (Codec.H265, None, 40000)


class VideoEncoder(Node, InputOutput):
    """High-level node for encoding video streams."""

    _inputs = ["input"]

    def __init__(
        self,
        pm: PipelineManager,
        filename: Path,
        econding_config: EncodingConfig,
        cam: Camera,
        add_codec_to_filename=True,
    ):
        """Create high-level video encoder node

        Args:
            pm (PipelineManager): associated pipeline manager
            filename (Path): file to write video stream to.
            econding_config (EncodingConfig): configuration for video encoder.
            cam (Camera): camera used for encoding.
            add_codec_to_filename (bool, optional): adds the codec as extension (e.g. '.h264'). Defaults to True.
        """
        super().__init__(pm)
        self.encoding_config = econding_config
        self.cam = cam
        self.filename = filename
        self.add_codec_to_filename = add_codec_to_filename

        # create encoder
        # Create XLinkOutputs for the stream
        pipeline = pm.pipeline
        self.xout = xout = pipeline.createXLinkOut()
        self.output_stream_name = self.pm.add_xstream_name("encoder")
        self.output_stream = None
        xout.setStreamName(self.output_stream_name)

        self.encoder = encoder = pipeline.createVideoEncoder()
        self.encoding_config.apply(encoder, cam.fps)

        self.cam.linkTo(self, cam.get_default_output(), "input")
        encoder.bitstream.link(xout.input)

        self.queue = None

    def activate(self, device: dai.Device):
        if self.add_codec_to_filename:
            self.filename = self.filename.with_suffix(
                "." + self.encoding_config.codec.name.lower()
            )

        self.queue = device.getOutputQueue(
            name=self.output_stream_name, maxSize=4, blocking=False
        )

        self.output_stream = open(self.filename, "wb")

    def deactivate(self):
        self.output_stream.close()

    def get(self):
        """
        Returns:
            byte: the encoding blob (!no raw image!)
        """
        return self.queue.get().getCvFrame()

    def write(self):
        """Writes the next stream blob to file"""
        data = self.get()

        self.output_stream.write(data)

    @property
    def inputs(self) -> list[str]:
        return self._inputs

    def get_input(self, name: str):
        assert name in self.inputs
        return getattr(self.encoder, name)

    @property
    def outputs(self) -> list[str]:
        return []

    def get_output(self, name: str):
        raise ValueError("No output endpoints available!")


class DepthEncoding(Node):
    """Depth encoding node"""


class LiveView(Node):
    """High-level node for live viewing image content"""

    def __init__(
        self,
        pm: PipelineManager,
        node: Node,
        output_name: str = None,
        data_modifier=None,
        max_queue_size=4,
    ):
        """Compose live view

        Args:
            pm (PipelineManager): The global pipeline manager.
            stream_name (str): The depthai pipeline stream name for the image stream.
            data_modifier (_type_): The data modifier that extracts cv images from depthai pacakges.
            maxQueueSize (int): max number of queued packages. Default: 4.
        """
        super().__init__(pm)
        self.data_modifier = data_modifier
        self.max_queue_size = max_queue_size
        self.qView = None
        self.node = node

        if output_name is None:
            if len(self.node.outputs) == 1:
                output_name = self.node.outputs[0]
            else:
                raise ValueError("Please specify an output name!")

        self.stream_name = output_name

        # TODO: this section is ugly. Should be handled in the nodes (ColorCamera, StereoDepth and MonoCamera)
        if data_modifier is None:
            assert hasattr(
                node, "get_data_modifier"
            ), "Please specify a data modifier as node does not support retrieval"
            self.data_modifier = node.get_data_modifier(output_name)

        assert (
            self.data_modifier
        ), "please specify a modifier to convert depthai data package to an opencv image"

        assert not output_name is None

        # create an output to host for the node output stream
        self.x_out = self.pm.pipeline.createXLinkOut()
        self.stream_name = self.pm.add_xstream_name(output_name + "_out")  # unique name
        self.x_out.setStreamName(self.stream_name)
        self.node.get_output(output_name).link(self.x_out.input)

    def activate(self, device):
        """Activates this nodes and starts all the device specific processes.

        Args:
            device (_type_): depthai device.
        """
        # open output queue on the device
        self.qView = device.getOutputQueue(
            name=self.stream_name, maxSize=self.max_queue_size, blocking=False
        )

    def get(self):
        """Retrieve an image frame from the xLink stream (blocking)

        Returns:
            np.array: cv frame
        """
        inFrame = self.qView.get()
        return self.data_modifier(inFrame)

    def try_get(self):
        """Retrieve an data frame from the xLink stream (non-blocking)

        Returns:
            any: data frame modified by the data modifier or None if there is no data frame available
        """
        inFrame = self.qView.tryGet()
        return self.data_modifier(inFrame)

    def show(self, frame_title="frame"):
        """Visualizes the current image frame from the stream

        Args:
            frame_title (str, optional): Title of the cv view. Defaults to "frame".
        """
        frame = self.get()

        cv2.imshow(frame_title, frame)


class FPSMeasure(Node):
    """fps measuring node"""


class ImageStream(Node):
    """image stream node"""


class MultiStreamRecorder(Node):
    """High-level node for recording multiple camera streams into compressed streams"""

    def __init__(
        self,
        pm: PipelineManager,
        nodes: dict[str, InputOutput],
        path=Path("recordings"),
        quality: EncodingConfig | dict[str, EncodingConfig] = EncodingConfig.LOW,
        fps=30,
    ):
        """

        Args:
            pm (PipelineManager): current pipeline manager
            nodes (Dict[str, InputOutput]): discribing the camera streams to save with (filename, camera object) format.
            path (_type_, optional): folder for stored recordings. Will be create if not existing. Defaults to Path("recordings").
            quality (EncodingConfig | Dict[str, EncodingConfig], optional): Either EncodingConfig that is applied to every camera stream or a config per stream (stream name, config). Defaults to EncodingConfig.LOW (low quality for all streams).
        """
        super().__init__(pm)

        self.fps = fps
        self.quality = EncodingConfig.HIGH
        self.path = path
        self.nodes = nodes
        self.quality = quality

        # check wheter a single config is applied
        if isinstance(quality, EncodingConfig):
            # copy the config for every node
            temp_quality = {}
            for node_name in nodes.keys():
                temp_quality[node_name] = quality

            self.quality = temp_quality

        self.encoders: dict[str, VideoEncoder] = {}
        for node_name, node in self.nodes.items():
            self.encoders[node_name] = VideoEncoder(
                pm, path / node_name, self.quality[node_name], node, True
            )

        self.device = None
        self.mxid = None

    def create_folder(self, path: Path, mxid: str) -> Path:
        """Creates a folder based on a path and the oak-d device id

        Args:
            path (Path): proposed path
            mxid (str): device id

        Returns:
            Path: the unique and existing path object
        """
        i = 0
        while True:
            i += 1
            recordings_path = path / f"{i}-{str(mxid)}"
            if not recordings_path.is_dir():
                recordings_path.mkdir(parents=True, exist_ok=False)
                return recordings_path

    def activate(self, device: dai.Device):
        self.device = device

        # create the path with device id
        self.mxid = device.getMxId()
        self.path = self.create_folder(self.path, self.mxid)

        # update encoder paths
        for node_name, encoder in self.encoders.items():
            encoder.filename = self.path / node_name

        # store calibration data
        calibData = device.readCalibration()
        calibData.eepromToJsonFile(str(self.path / "calib.json"))

    def write(self):
        """Store all encoders to files"""
        for encoder in self.encoders.values():
            encoder.write()


class SystemLogger(Node, InputOutput):
    """Node to display OAK-D system information (e.g. memory usage, temperature)"""

    def __init__(self, pm: PipelineManager, rate=1, obtain=True):
        """Create a new system logger node

        Args:
            pm (PipelineManager): the global pipeline manager.
            rate (float, optional): Rate in Hz to retrieve system information. Defaults to 1.
            obtain (bool, optional): whether to connect to the pipeline node . Defaults to True.
        """
        super().__init__(pm)

        # create system logger node
        self.sysLog = pm.pipeline.create(dai.node.SystemLogger)
        # set sampling rate
        self.sysLog.setRate(rate)

        # configure node outputs
        self._outputs = {"out": self.sysLog.out}

        # create view on data if needed
        if obtain:
            self.lv = LiveView(pm, self, "out")

    @property
    def inputs(self) -> list[str]:
        return []

    def get_input(self, name: str):
        raise ValueError("No inputs available!")

    @property
    def outputs(self) -> list[str]:
        return self._outputs.keys()

    def get_output(self, name: str):
        assert name in self.outputs
        return self._outputs[name]

    def get_data_modifier(self, name: str):
        if name in self.outputs:
            # just identity, we do not need to modify the data
            return lambda data: data
        raise ValueError("Unkwnown output name")

    def print_system_information(self, information=None):
        """Print system information to stdout if available

        Args:
            information (_type_, optional): A specific information package that we want to print. If None, we try to obtain new information. Defaults to None.
        """
        if information is None:
            # get new information
            information = self.get_system_information()

        if information is not None:
            # print when info is available
            self.printSystemInformation(information)

    def get_system_information(self):
        """Obtain system information form xLink

        Returns:
            _type_: system information package or None (if not yet available)
        """

        return self.lv.try_get()

    @staticmethod
    def printSystemInformation(info):
        """Print system information in stdout

        Args:
            info (_type_): system information
        """
        m = 1024 * 1024  # MiB
        print(
            f"Ddr used / total - {info.ddrMemoryUsage.used / m:.2f} / {info.ddrMemoryUsage.total / m:.2f} MiB"
        )
        print(
            f"Cmx used / total - {info.cmxMemoryUsage.used / m:.2f} / {info.cmxMemoryUsage.total / m:.2f} MiB"
        )
        print(
            f"LeonCss heap used / total - {info.leonCssMemoryUsage.used / m:.2f} / {info.leonCssMemoryUsage.total / m:.2f} MiB"
        )
        print(
            f"LeonMss heap used / total - {info.leonMssMemoryUsage.used / m:.2f} / {info.leonMssMemoryUsage.total / m:.2f} MiB"
        )
        t = info.chipTemperature
        print(
            f"Chip temperature - average: {t.average:.2f}, css: {t.css:.2f}, mss: {t.mss:.2f}, upa: {t.upa:.2f}, dss: {t.dss:.2f}"
        )
        print(
            f"Cpu usage - Leon CSS: {info.leonCssCpuUsage.average * 100:.2f}%, Leon MSS: {info.leonMssCpuUsage.average * 100:.2f} %"
        )
        print("----------------------------------------")
