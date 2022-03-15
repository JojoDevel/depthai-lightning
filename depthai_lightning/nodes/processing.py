""" High-level data processing nodes"""

import json
from abc import ABC
from pathlib import Path
from typing import List

import cv2
import depthai as dai
import numpy as np

from depthai_lightning.depthai_lightning import PipelineManager
from depthai_lightning.nodes.modifiers import preview_modifier
from depthai_lightning.nodes.output import LiveView

from .base import InputOutput, Node
from .input import ColorCamera, MonoCamera


class ObjectDetector(Node, InputOutput, ABC):
    """Object detector node"""


class YoloDetector(ObjectDetector):
    """Object detector using yolo model"""

    def __init__(
        self,
        pm: PipelineManager,
        config: Path,
        blob: Path,
        camRgb: ColorCamera,
        provide_rgb=True,
    ):
        super().__init__(pm)

        self.provide_rgb = provide_rgb

        assert config.exists()
        assert blob.exists()

        with open(config, encoding="utf-8") as config_input:
            yolo_config = json.load(config_input)

        nnConfig = yolo_config.get("nn_config", {})

        # parse input shape
        if "input_size" in nnConfig:
            W, H = tuple(map(int, nnConfig.get("input_size").split("x")))

        # extract metadata
        metadata = nnConfig.get("NN_specific_metadata", {})
        classes = metadata.get("classes", {})
        coordinates = metadata.get("coordinates", {})
        anchors = metadata.get("anchors", {})
        anchorMasks = metadata.get("anchor_masks", {})
        iouThreshold = metadata.get("iou_threshold", {})
        confidenceThreshold = metadata.get("confidence_threshold", {})

        print(metadata)

        # parse labels
        nnMappings = yolo_config.get("mappings", {})
        self.labels = nnMappings.get("labels", {})

        # sync outputs
        # syncNN = True

        detectionNetwork = pm.pipeline.create(dai.node.YoloDetectionNetwork)

        assert camRgb.preview_size == (
            W,
            H,
        ), f"Yolo network expects input size ({W}, {H}) but camera gives ({camRgb.preview_size[0]}, {camRgb.preview_size[1]})"
        camRgb.cam.setInterleaved(False)

        # Network specific settings
        detectionNetwork.setConfidenceThreshold(confidenceThreshold)
        detectionNetwork.setNumClasses(classes)
        detectionNetwork.setCoordinateSize(coordinates)
        detectionNetwork.setAnchors(anchors)
        detectionNetwork.setAnchorMasks(anchorMasks)
        detectionNetwork.setIouThreshold(iouThreshold)
        detectionNetwork.setBlobPath(blob)
        detectionNetwork.setNumInferenceThreads(2)
        detectionNetwork.input.setBlocking(False)

        # define inputs and outputs
        self._inputs = {"input": detectionNetwork.input}
        self._outputs = {
            "rgb": detectionNetwork.passthrough,
            "passthrough": detectionNetwork.passthrough,
            "out": detectionNetwork.out,
        }

        if self.provide_rgb:
            self.lv_rgb = LiveView(pm, self, "rgb", preview_modifier)
            self.lv_nn = LiveView(pm, self, "out", lambda d: d)

        # Linking
        camRgb.linkTo(self, "preview", "input")

        # Cache for detections and frame
        self.last_detections = None
        self.last_frame = None

    @property
    def inputs(self) -> List[str]:
        return self._inputs.keys()

    def get_input(self, name: str):
        assert name in self._inputs
        return self._inputs[name]

    @property
    def outputs(self) -> List[str]:
        return self._outputs.keys()

    def get_output(self, name: str):
        assert name in self.outputs
        return self._outputs[name]

    def activate(self, device: dai.Device):
        # all preparations are performed by high-level nodes
        pass

    def get_detections(self):
        """Retrieve brand-new detections package from OAK-D devie

        Returns:
            _type_: detections package
        """
        self.last_detections = self.lv_nn.get()
        return self.last_detections

    def get_frame(self):
        """Retrieve brand-new frame from camera stream

        Returns:
            np.arry: cv2 image
        """
        assert self.provide_rgb, "You need to active rgb output to access the frame"
        self.last_frame = self.lv_rgb.get()
        return self.last_frame

    @staticmethod
    def frameNorm(frame, bbox):
        """Scale bbox values [0...1] to frame size

        Args:
            frame (np.array): camera frame
            bbox (np.arry): bbox coordinates for detections

        Returns:
            np.arry: bbox coordinates with pixel coordinates
        """

        # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayFrame(self, name: str, frame=None, new_detections=True):
        """Displays the video frame and draws detection results

        Args:
            name (str): Name of the cv2 window.
            frame (_type_, optional): The frame of the video feed. Defaults to None (we get the frame internally).
            new_detections (bool, optional): Retrieve brand-new detections from the device or used cached ones. Defaults to True.
        """
        if new_detections:
            # get brand-new detections
            detections = self.get_detections().detections
            if frame is None:
                # and also frame
                frame = self.get_frame()
        else:
            # get cached detections
            detections = self.last_detections.detections
            if frame is None:
                # and also frame
                frame = self.last_frame

        # blue color for object bboxes
        color = (255, 0, 0)
        for detection in detections:
            # convert normalized coordinates to pixel coordinates
            bbox = YoloDetector.frameNorm(
                frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax)
            )
            # write label name
            cv2.putText(
                frame,
                self.labels[detection.label],
                (bbox[0] + 10, bbox[1] + 20),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                255,
            )
            # write confidence
            cv2.putText(
                frame,
                f"{int(detection.confidence * 100)}%",
                (bbox[0] + 10, bbox[1] + 40),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                255,
            )
            # draw bbox
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

        # Show the frame
        cv2.imshow(name, frame)


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

    def get_data_modifier(self, output_name):
        assert output_name in self.outputs

        return preview_modifier


class ImageCrop(Node):
    """image cropping node"""


class PythonNode(Node):
    """node for custom python node on-device"""
