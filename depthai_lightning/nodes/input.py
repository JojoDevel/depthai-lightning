""" High-level input nodes """
import os
from abc import ABC
from pathlib import Path
from typing import Dict, Tuple

import cv2
import depthai as dai

from depthai_lightning.depthai_lightning import PipelineManager
from depthai_lightning.nodes.modifiers import (
    isp_modifier,
    preview_modifier,
    raw_modifier,
)

from .base import InputOutput, Node


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


class Camera(Node, InputOutput, ABC):
    """Basic camera node"""

    def __init__(self, pm: PipelineManager, fps: int, default_output: str = None):
        """Create camera object

        Args:
            pm (PipelineManager): the pipeline manager for this camera
            fps (int): recording framerate
            default_output (str, optional): Name of the default output stream. Defaults to None.
        """
        super().__init__(pm)

        self._fps = fps
        self.default_output = default_output

    def get_default_output(self) -> str:
        """
        Returns:
            str: the name of the default output stream
        """
        return self.default_output

    @property
    def fps(self) -> int:
        return self._fps


class MonoCamera(Camera):
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
        super().__init__(pm, fps, MonoCamera.DEFAULT_OUT_STREAM)

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

    def get_data_modifier(self, output_name):
        assert output_name in self.outputs

        return preview_modifier


class ColorCamera(Camera):
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
        super().__init__(pm, fps, "video")

        self.cam = pm.pipeline.createColorCamera()
        self.resolution = resolution
        self.rotate = rotate
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
        return ["preview", "raw", "isp", "video"]

    @property
    def inputs(self):
        return []

    def get_input(self, name: str):
        raise ValueError(f"{self.__class__.__name__} does not have any inputs")

    def get_output(self, name: str):
        assert name in self.outputs

        return getattr(self.cam, name)

    def get_data_modifier(self, output_name):
        assert output_name in self.outputs

        return {
            "preview": preview_modifier,
            "raw": raw_modifier,
            "isp": isp_modifier,
            "video": preview_modifier,
        }[output_name]
