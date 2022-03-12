""" High-level output nodes """


from enum import Enum
from typing import List

import cv2
import depthai as dai

from depthai_lightning.depthai_lightning import PipelineManager

from .base import InputOutput, Node
from .input import Camera, ColorCamera, MonoCamera
from .modifiers import isp_modifier, preview_modifier, raw_modifier
from .processing import StereoDepth


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

    # some default configs
    BEST = (Codec.MJPEG, 100, None)  # Lossless MJPEG
    HIGH = (Codec.MJPEG, 97, None)  # MJPEG Quality=97 (default)
    MEDIUM = (Codec.MJPEG, 97, None)  # MJPEG Quality=93
    LOW = (Codec.H265, None, 10000)  # H265 BitrateKbps=10000


class VideoEncoder(Node, InputOutput):
    """High-level node for encoding video streams."""

    _inputs = ["input"]

    def __init__(
        self,
        pm: PipelineManager,
        filename: str,
        econding_config: EncodingConfig,
        cam: Camera,
        add_codec_to_filename=True,
    ):
        """Create high-level video encoder node

        Args:
            pm (PipelineManager): associated pipeline manager
            filename (str): file to write video stream to.
            econding_config (EncodingConfig): configuration for video encoder.
            cam (Camera): camera used for encoding.
            add_codec_to_filename (bool, optional): adds the codec as extension (e.g. '.h264'). Defaults to True.
        """
        super().__init__(pm)
        self.encoding_config = econding_config
        self.cam = cam
        self.filename = filename
        if add_codec_to_filename:
            self.filename += "." + self.encoding_config.codec.name.lower()

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
    def inputs(self) -> List[str]:
        return self._inputs

    def get_input(self, name: str):
        assert name in self.inputs
        return getattr(self.encoder, name)

    @property
    def outputs(self) -> List[str]:
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
        if data_modifier is None and isinstance(node, ColorCamera):
            if output_name in ["preview", "video"]:
                self.data_modifier = preview_modifier
            elif output_name == "isp":
                self.data_modifier = isp_modifier
            elif output_name == "raw":
                self.data_modifier = raw_modifier
        elif data_modifier is None and isinstance(node, StereoDepth):
            if output_name == "depth":
                self.data_modifier = preview_modifier
        elif data_modifier is None and isinstance(node, MonoCamera):
            if output_name == "out":
                self.data_modifier = preview_modifier

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
        """Retrieve an image frame from the stream

        Returns:
            np.array: cv frame
        """
        inFrame = self.qView.get()
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