"""All the high-level nodes
"""

from typing import List

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
            selected_input = self.get_input(target_input)
        else:
            if len(inputs) == 1:
                # choose default input
                selected_input = self.get_input(inputs[0])
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


class Camera(Node):
    """Basic camera node"""


class CameraColor(Camera):
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

        self.cam.setResolution(CameraColor.rgb_res_opts.get(self.resolution))
        self.cam.setPreviewSize(*self.preview_size)

        # Optional, set manual focus. 255: macro (8cm), about 120..130: infinity
        # if args.lens_position >= 0:
        #    cam.initialControl.setManualFocus(args.lens_position)
        # cam.setIspScale(1, args.isp_downscale)
        self.cam.setFps(self.fps)  # Default: 30
        if self.rotate:
            self.cam.setImageOrientation(dai.CameraImageOrientation.ROTATE_180_DEG)

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


class CameraGray(Camera):
    """Gray camera input node"""


## Processing
class ObjectDetector(Node):
    """Object detector node"""


class SpatialDetector(ObjectDetector):
    """Spatial detector node"""


class ObjectTracker(Node):
    """Object tracking node"""


class StereoDepth(Node):
    """stereo depth node"""


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
        self, pm: PipelineManager, stream_name: str, data_modifier, max_queue_size=4
    ):
        """Compose live view

        Args:
            pm (PipelineManager): The global pipeline manager.
            stream_name (str): The depthai pipeline stream name for the image stream.
            data_modifier (_type_): The data modifier that extracts cv images from depthai pacakges.
            maxQueueSize (int): max number of queued packages. Default: 4.
        """
        super().__init__(pm)
        self.stream_name = stream_name
        self.data_modifier = data_modifier
        self.max_queue_size = max_queue_size
        self.qView = None

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
