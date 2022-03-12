"""All the high-level nodes
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import depthai as dai
import numpy as np

from depthai_lightning import PipelineManager

from .utils import unpack_raw10


class InputOutput:
    """Provides abstract class for input and output linking

    Raises:
        NotImplementedError: input/output endpoints have to be implemented by subclasses.
        ValueError: when linking fails due to wrong input/output names
    """

    @property
    def inputs(self) -> List[str]:
        """List all possible input names.

        These input names can be used to fill their input with the output of another node.

        Returns:
            List[str]: list of all input names.
        """
        raise NotImplementedError()

    @property
    def outputs(self) -> List[str]:
        """List all possible output names.

        These output names can be used to fill inputs of other nodes

        Returns:
            List[str]: list of all output names
        """
        raise NotImplementedError()

    def get_input(self, name: str):
        """get the depthai object of an input name.

        Args:
            name (str): input name

        Raises:
            NotImplementedError: Please implement this abstract method
        """
        raise NotImplementedError()

    def get_output(self, name: str):
        """get the depthai object of an output name.

        Args:
            name (str): output name

        Raises:
            NotImplementedError: Please implement this abstract method
        """
        raise NotImplementedError()

    def __getitem__(self, name: str):
        """get input or output name

        Args:
            name (str): input or output name

        Raises:
            ValueError: if name does not match any input or output name

        Returns:
            _type_: _description_
        """
        if name in self.inputs:
            return self.get_input(name)

        if name in self.outputs:
            return self.get_output(name)

        raise ValueError(f"{name} is neither input or output of this node.")

    def linkTo(self, target: "Node", my_output: str = None, target_input: str = None):
        """Links an output of this node to an input of another node

        Args:
            target (Node): target node
            my_output (str, optional): this nodes output name. Defaults to None (using default output).
            target_input (str, optional): the target input name. Defaults to None (using default input).

        Raises:
            ValueError: when default input/output is not available (multiple possibilities) and no specifc value has been specified.
        """
        inputs = target.inputs
        outputs = self.outputs

        selected_output = None
        selected_input = None
        if my_output:
            selected_output = self.get_output(my_output)
        else:
            if len(outputs) == 1:
                # choose default output
                selected_output = self.get_output(outputs[0])
            else:
                raise ValueError(
                    f"Multiple outputs available {outputs} but none is specified!"
                )

        if target_input:
            selected_input = target.get_input(target_input)
        else:
            if len(inputs) == 1:
                # choose default input
                selected_input = target.get_input(inputs[0])
            else:
                raise ValueError(
                    f"Multiple outputs available {outputs} but none is specified!"
                )

        assert selected_output and selected_input

        # link my output to target's input
        selected_output.link(selected_input)


class Node:
    """Base class for high-level nodes"""

    def __init__(self, pm: PipelineManager):
        self.pm = pm

        self.pm.nodes.append(self)

    def activate(self, device: dai.Device):
        pass


## input
class Photo(Node):
    """Photo input node"""


class Video(Node):
    """Video Input node"""


class Replay(Node):
    """Node to replay recorded material into the OAK-D device.

    Includes code from https://github.com/luxonis/depthai-experiments/tree/master/gen2-record-replay
    """

    def __init__(self, pm: PipelineManager, path: str):
        """Create a replay node

        Args:
            pm (PipelineManager): the global pipeline manager.
            path (str): path to the folder containing the stream recordings
        """
        super().__init__(pm)

        self.path = Path(path).resolve().absolute()

        self.cap = {}  # VideoCapture objects
        self.size = {}  # Frame sizes
        self.frames = {}  # Frames read from the VideoCapture
        self.queues = {}

        self.nodes = {}

        # Disparity shouldn't get streamed to the device, nothing to do with it.
        self.stream_types = ["color", "left", "right", "depth"]

        file_types = ["color", "left", "right", "disparity", "depth"]
        extensions = ["mjpeg", "avi", "mp4", "h265", "h264"]

        for file in os.listdir(path):
            if not "." in file:
                continue  # Folder
            name, extension = file.split(".")
            if name in file_types and extension in extensions:
                self.cap[name] = cv2.VideoCapture(str(self.path / file))

        if len(self.cap) == 0:
            raise RuntimeError("There are no recordings in the folder specified.")

        # Load calibration data from the recording folder
        self.calibData = dai.CalibrationHandler(str(self.path / "calib.json"))

        # Read basic info about the straems (resolution of streams etc.)
        for name, cap in self.cap.items():
            self.size[name] = self.get_size(cap)

        self.color_size: Tuple[int, int] = None

        # keep the aspect ratio
        self.keep_ar = True

        self.init_pipeline()

    @property
    def outputs(self):
        return self.nodes.keys()

    def get_output(self, name: str):
        assert name in self.nodes

        return self.nodes[name]

    @property
    def left(self) -> InputOutput:
        """We are mimicing the mono behavior

        Returns:
            InputOutput: node mimicing a mono camer input/output
        """
        assert "left" in self.nodes, "Missing left camera stream!"

        return StreamWrapper(
            input_streams={}, output_streams={"out": self.nodes["left"].out}
        )

    @property
    def right(self) -> InputOutput:
        """We are mimicing the mono behavior here

        Returns:
            InputOutput: node mimicing a mono camera input/output
        """

        assert "right" in self.nodes, "Missing right camera stream!"

        return StreamWrapper(
            input_streams={}, output_streams={"out": self.nodes["right"].out}
        )

    def create_queues(self, device):
        self.queues = {}
        for name in self.cap:
            if name in self.stream_types:
                self.queues[name + "_in"] = device.getInputQueue(name + "_in")

    def activate(self, device: dai.Device):
        self.create_queues(device)

    def perform(self):
        self.send_frames()

    def get_size(self, cap):
        return (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )

    def get_max_size(self, name):
        total = self.size[name][0] * self.size[name][1]
        if name == "color":
            total *= 3  # 3 channels
        return total

    def send_frames(self) -> bool:
        """Read all frames and send them via xlink to OAK-D.

        Returns:
            bool: True when successful, False when errors occur
        """
        if self.read_frames():
            return False  # end of recording
        for name, frame in self.frames.items():
            if name in ["left", "right", "disparity"] and len(frame.shape) == 3:
                self.frames[name] = frame[:, :, 0]  # All 3 planes are the same

            self.send_frame(self.frames[name], name)

        return True

    def create_stream(self, name: str):
        pipeline = self.pm.pipeline
        node = pipeline.createXLinkIn()
        node.setMaxDataSize(self.get_max_size(name))
        stream_name = f"{name}_in"
        node.setStreamName(stream_name)

        return node

    def init_pipeline(self):
        pipeline = self.pm.pipeline

        mono = ("left" in self.cap) and ("right" in self.cap)
        depth = "depth" in self.cap

        pipeline.setCalibrationData(self.calibData)

        if "color" in self.cap:
            # create color stream
            self.nodes["color"] = self.create_stream("color")

        if mono:
            # create mono streams
            self.nodes["left"] = self.create_stream("left")
            self.nodes["right"] = self.create_stream("right")

            # stereo_node = pipeline.createStereoDepth()
            # stereo_node.setInputResolution(self.size['left'][0], self.size['left'][1])

            # self.nodes['left'].out.link(stereo_node.left)
            # self.nodes['right'].out.link(stereo_node.right)

            # self.nodes['stereo'] = stereo_node

        if depth:
            # create depth stream
            depth_node = pipeline.createXLinkIn()
            depth_node.setStreamName("depth_in")
            depth_node.setSubpixel(True)

            depth_node.setDefaultProfilePreset(
                dai.node.StereoDepth.PresetMode.HIGH_DENSITY
            )
            depth_node.initialConfig.setMedianFilter(
                dai.MedianFilter.MEDIAN_OFF
            )  # KERNEL_7x7 default
            depth_node.setRectifyEdgeFillColor(0)  # Black, to better see the cutout
            depth_node.setLeftRightCheck(False)
            depth_node.setExtendedDisparity(False)
            depth_node.setSubpixel(True)

            depth_node.initialConfig.setMedianFilter(dai.MedianFilter.MEDIAN_OFF)

            depth_node.setDepthAlign(dai.CameraBoardSocket.RGB)

            # depth_node.setConfidenceThreshold(255)

            self.nodes["depth"] = depth_node

    def to_planar(self, arr, shape=None):
        if shape is not None:
            arr = cv2.resize(arr, shape)
        return arr.transpose(2, 0, 1).flatten()

    def read_frames(self) -> bool:
        """Read next frames from the streams.

        Returns:
            bool: returns True when frames have been read, False otherwise
        """
        self.frames = {}
        # for every open stream -> read frame
        for name, cap in self.cap.items():
            if not cap.isOpened():
                return True
            ok, frame = cap.read()
            if ok:
                self.frames[name] = frame
        return len(self.frames) == 0

    def send_frame(self, frame, name: str):
        """Send frame to OAK-D

        Args:
            frame (np.array): raw frame data
            name (str): name of the stream
        """
        q_name = name + "_in"
        if q_name in self.queues:
            if name == "color":
                # self.send_color(self.queues[q_name], frame)
                pass
            elif name == "left":
                self.send_mono(self.queues[q_name], frame, False)
            elif name == "right":
                self.send_mono(self.queues[q_name], frame, True)
            elif name == "depth":
                self.send_depth(self.queues[q_name], frame)

    def send_mono(self, q, img, right):
        self.frames["right" if right else "left"] = img
        h, w = img.shape
        frame = dai.ImgFrame()
        frame.setData(img)
        frame.setType(dai.RawImgFrame.Type.RAW8)
        frame.setWidth(w)
        frame.setHeight(h)
        frame.setInstanceNum(2 if right else 1)
        q.send(frame)

    def send_color(self, q, img):
        # Resize/crop color frame as specified by the user
        img = self.resize_color(img)
        self.frames["color"] = img
        h, w, _ = img.shape
        frame = dai.ImgFrame()
        frame.setType(dai.RawImgFrame.Type.BGR888p)
        frame.setData(self.to_planar(img))
        frame.setWidth(w)
        frame.setHeight(h)
        frame.setInstanceNum(0)
        q.send(frame)

    def send_depth(self, q, depth):
        # TODO refactor saving depth. Reading will be from ROS bags.

        # print("depth size", type(depth))
        # depth_frame = np.array(depth).astype(np.uint8).view(np.uint16).reshape((400, 640))
        # depthFrameColor = cv2.normalize(depth_frame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        # depthFrameColor = cv2.equalizeHist(depthFrameColor)
        # depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)
        # cv2.imshow("depth", depthFrameColor)
        frame = dai.ImgFrame()
        frame.setType(dai.RawImgFrame.Type.RAW16)
        frame.setData(depth)
        frame.setWidth(640)
        frame.setHeight(400)
        frame.setInstanceNum(0)
        q.send(frame)

    def resize_color(self, frame):
        if self.color_size is None:
            # No resizing needed
            return frame

        if not self.keep_ar:
            # No need to keep aspect ratio, image will be squished
            return cv2.resize(frame, self.color_size)

        h = frame.shape[0]
        w = frame.shape[1]
        desired_ratio = self.color_size[0] / self.color_size[1]
        current_ratio = w / h

        # Crop width/heigth to match the aspect ratio needed by the NN
        if desired_ratio < current_ratio:  # Crop width
            # Use full height, crop width
            new_w = (desired_ratio / current_ratio) * w
            crop = int((w - new_w) / 2)
            preview = frame[:, crop : w - crop]
        else:  # Crop height
            # Use full width, crop height
            new_h = (current_ratio / desired_ratio) * h
            crop = int((h - new_h) / 2)
            preview = frame[crop : h - crop, :]

        return cv2.resize(preview, self.color_size)


class StreamWrapper(InputOutput):
    """Class to rename streams

    For example: You have an object that needs to access stream 'out' but you current object only provides 'cam'. With this node you can wrap the streams so that 'out' -> 'cam'.
    """

    def __init__(self, input_streams: Dict[str, any], output_streams: Dict[str, any]):
        """Create wrapping for streams

        Args:
            input_streams (Dict[str, stream]): Renaming of input streams (input new name, input stream)
            output_streams (Dict[str, stream]): Renaming of output streams (output new name, output stream)
        """
        self.input_streams = input_streams
        self.output_streams = output_streams

    @property
    def outputs(self):
        return self.output_streams.keys()

    def get_output(self, name: str):
        return self.output_streams[name]

    @property
    def inputs(self):
        return self.input_streams.keys()

    def get_input(self, name: str):
        return self.input_streams[name]


class Camera(Node):
    """Basic camera node"""


class MonoCamera(Camera, InputOutput):
    """Mono Camera implementation"""

    DEFAULT_OUT_STREAM = "out"

    res_mapping = {
        800: dai.MonoCameraProperties.SensorResolution.THE_800_P,
        720: dai.MonoCameraProperties.SensorResolution.THE_720_P,
        480: dai.MonoCameraProperties.SensorResolution.THE_480_P,
        400: dai.MonoCameraProperties.SensorResolution.THE_400_P,
    }

    def __init__(
        self,
        pm: PipelineManager,
        socket: dai.CameraBoardSocket,
        res_lower: int,
        fps: int = 30,
    ):
        super().__init__(pm)

        self.fps = fps
        self.socket = socket
        self.res = res_lower

        assert res_lower in self.res_mapping

        # create pipeline node
        self.cam = pm.pipeline.create(dai.node.MonoCamera)

        # set socket
        self.cam.setBoardSocket(socket)

        # resolution
        self.cam.setResolution(self.res_mapping[res_lower])
        self.cam.setFps(self.fps)

    @property
    def inputs(self):
        # this node has no inputs
        return []

    def get_input(self, name: str):
        raise ValueError("No inputs available!")

    @property
    def outputs(self):
        return [self.DEFAULT_OUT_STREAM]

    def get_output(self, name: str):
        assert name in self.outputs, f"Unkwown output stream {name}"

        if name == self.DEFAULT_OUT_STREAM:
            return self.cam.out

        raise ValueError(f"Do not know output stream {name}")


class ColorCamera(Camera, InputOutput):
    """High-level node for the depthai camera"""

    rgb_res_opts = {
        "1080": dai.ColorCameraProperties.SensorResolution.THE_1080_P,
        "4k": dai.ColorCameraProperties.SensorResolution.THE_4_K,
        "12mp": dai.ColorCameraProperties.SensorResolution.THE_12_MP,
    }

    def __init__(
        self,
        pm: PipelineManager,
        name="colorCam",
        resolution="1080",
        fps=30,
        rotate=False,
        preview_size=(300, 300),
        isp_scale=(1, 1),
    ):
        """Create a depthai color camera

        Args:
            pm (PipelineManager): The pipeline manager.
            name (str): The depthai base name.
            resolution (str, optional): Recording resolution. Defaults to '1080'.
            fps (int, optional): Recording framerate. Defaults to 30.
            rotate (bool, optional): Rotate camera image by 180 degrees. Defaults to False.
            preview_size (Tuple[int, int]): size of the camera preview.
        """
        super().__init__(pm)

        self.cam = pm.pipeline.createColorCamera()
        self.resolution = resolution
        self.rotate = rotate
        self.fps = fps
        self.preview_size = preview_size
        self.name = name
        self.isp_scale = isp_scale

        self.cam.setBoardSocket(dai.CameraBoardSocket.RGB)
        self.cam.setResolution(ColorCamera.rgb_res_opts.get(self.resolution))
        self.cam.setPreviewSize(*self.preview_size)
        self.cam.setIspScale(*self.isp_scale)

        # Optional, set manual focus. 255: macro (8cm), about 120..130: infinity
        # if args.lens_position >= 0:
        #    cam.initialControl.setManualFocus(args.lens_position)
        # cam.setIspScale(1, args.isp_downscale)
        self.cam.setFps(self.fps)  # Default: 30
        if self.rotate:
            self.cam.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)

    @property
    def outputs(self):
        return ["preview", "raw", "isp"]

    @property
    def inputs(self):
        return []

    def get_input(self, name: str):
        raise ValueError(f"{self.__class__.__name__} does not have any inputs")

    def get_output(self, name: str):
        assert name in self.outputs

        return getattr(self.cam, name)

    def liveView(self, stream="preview"):
        """Create a live view object for a specific camera stream preview, raw or isp.

        Args:
            stream (str, optional): camera stream to view ('preview', 'raw' or 'isp'). Defaults to 'preview'.

        Returns:
            LiveView: live viewing node for the specified camera stream
        """
        modifier = None

        # compose depthai streams according to the requested camera stream
        if stream == "preview":
            stream_name = f"{self.name}_prev"
            x_out_preview = self.pm.pipeline.createXLinkOut()
            x_out_preview.setStreamName(stream_name)
            self.cam.preview.link(x_out_preview.input)
            modifier = preview_modifier
        elif stream == "raw":
            stream_name = f"{self.name}_raw"
            x_out_raw = self.pm.pipeline.createXLinkOut()
            x_out_raw.setStreamName(stream_name)
            self.cam.raw.link(x_out_raw.input)
            modifier = raw_modifier
        elif stream == "isp":
            stream_name = f"{self.name}_isp"
            x_out_isp = self.pm.pipeline.createXLinkOut()
            x_out_isp.setStreamName(stream_name)
            self.cam.isp.link(x_out_isp.input)
            modifier = isp_modifier
        else:
            raise ValueError(f"Cannot handle live view for strem {stream}")

        # compose and return the live view object
        return LiveView(self.pm, stream_name, data_modifier=modifier)


## Processing
class ObjectDetector(Node):
    """Object detector node"""


class SpatialDetector(ObjectDetector):
    """Spatial detector node"""


class ObjectTracker(Node):
    """Object tracking node"""


class StereoConfig:
    """Configuration class for stereo node. Collects all the static initial configuration options"""

    def __init__(
        self,
        rectify_edge_fill_color=0,
        left_right_check=False,
        subpixel=False,
        extended_disparity=False,
        default_profile_preset=dai.node.StereoDepth.PresetMode.HIGH_DENSITY,
        median_filter=dai.StereoDepthProperties.MedianFilter.KERNEL_3x3,
        depth_align=dai.CameraBoardSocket.RGB,
        input_resolution=None,
    ):

        self.rectify_edge_fill_color = rectify_edge_fill_color
        self.left_right_check = left_right_check
        self.subpixel = subpixel
        self.extended_disparity = extended_disparity
        self.default_profile_preset = default_profile_preset
        self.median_filter = median_filter
        self.depth_align = depth_align

        self.input_resolution = input_resolution

        assert not (
            subpixel and extended_disparity
        ), "Subpixel and ExtendedDisparity are not supported at the same time"


class StereoDepth(Node):
    """stereo depth node"""

    _inputs = ["left", "right", "inputConfig"]
    _outputs = [
        "syncedLeft",
        "syncedRight",
        "disparity",
        "depth",
        "rectifiedLeft",
        "rectifiedRight",
        "confidenceMap",
        "outConfig",
    ]

    def __init__(
        self,
        pm: PipelineManager,
        leftCam: MonoCamera,
        rightCam: MonoCamera,
        config=StereoConfig(),
    ):
        super().__init__(pm)
        self.config = config

        self.stereo = self.pm.pipeline.create(dai.node.StereoDepth)

        self.leftCam = leftCam
        self.rightCam = rightCam

        # linking outputs of mono cameras to the stereo node
        leftCam.linkTo(self, MonoCamera.DEFAULT_OUT_STREAM, "left")
        rightCam.linkTo(self, MonoCamera.DEFAULT_OUT_STREAM, "right")

        # configure stereo node
        self.stereo.setRectifyEdgeFillColor(config.rectify_edge_fill_color)
        self.stereo.setSubpixel(config.subpixel)
        self.stereo.setExtendedDisparity(config.extended_disparity)
        self.stereo.setDefaultProfilePreset(config.default_profile_preset)
        self.stereo.initialConfig.setMedianFilter(config.median_filter)
        self.stereo.setDepthAlign(config.depth_align)

        if config.input_resolution:
            self.stereo.setInputResolution(*config.input_resolution)

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs

    def get_input(self, name: str):
        assert name in self.inputs
        return getattr(self.stereo, name)

    def get_output(self, name: str):
        assert name in self.outputs
        return getattr(self.stereo, name)


class ImageCrop(Node):
    """image cropping node"""


class PythonNode(Node):
    """node for custom python node on-device"""


## Output


class StreamEncoding(Node):
    """Video stream encoding node"""


class DepthEncoding(Node):
    """Depth encoding node"""


def preview_modifier(inFrame):
    """Convert preview data package to cv frame

    Args:
        inFrame (_type_): depthai data package

    Returns:
        np.array: cv frame in bgr
    """
    return inFrame.getCvFrame()


def isp_modifier(inFrame):
    """Convert isp data package to cv frame

    Args:
        inFrame (_type_): depthai data package

    Returns:
        _type_: cv frame in bgr
    """
    width, height = inFrame.getWidth(), inFrame.getHeight()
    payload = inFrame.getData()

    # isp specific handling
    shape = (height * 3 // 2, width)
    yuv420p = payload.reshape(shape).astype(np.uint8)
    bgr = cv2.cvtColor(yuv420p, cv2.COLOR_YUV2BGR_IYUV)

    return bgr


def raw_modifier(inFrame):
    """Convert raw data package to cv frame

    Args:
        inFrame (_type_): depthai data package

    Returns:
        _type_:
    """
    width, height = inFrame.getWidth(), inFrame.getHeight()
    payload = inFrame.getData()

    # Preallocate the output buffer
    unpacked = np.empty(payload.size * 4 // 5, dtype=np.uint16)
    # Full range for display, use bits [15:6] of the 16-bit pixels
    unpack_raw10(payload, unpacked, expand16bit=True)
    shape = (height, width)
    bayer = unpacked.reshape(shape).astype(np.uint16)
    # See this for the ordering, at the end of page:
    # https://docs.opencv.org/4.5.1/de/d25/imgproc_color_conversions.html
    bgr = cv2.cvtColor(bayer, cv2.COLOR_BayerBG2BGR)

    return bgr


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
            if output_name == "preview":
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
        ), "please a modifier to convert depthai data package to an opencv image"

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
