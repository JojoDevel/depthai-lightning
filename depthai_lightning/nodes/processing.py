""" High-level data processing nodes"""
from __future__ import annotations

import json
from abc import ABC
from pathlib import Path

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
        self.camRgb = camRgb

        assert config.exists()
        assert blob.exists()

        with open(config, encoding="utf-8") as config_input:
            self.yolo_config = json.load(config_input)

        self.__create_detection_network(blob)

        # sync outputs
        # syncNN = True

        # Cache for detections and frame
        self.last_detections = None
        self.last_frame = None

    def __create_detection_network(self, blob):
        nnConfig = self.yolo_config.get("nn_config", {})

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
        nnMappings = self.yolo_config.get("mappings", {})
        self.labels = nnMappings.get("labels", {})

        detectionNetwork = self.pm.pipeline.create(dai.node.YoloDetectionNetwork)

        if isinstance(self.camRgb, ColorCamera):
            assert self.camRgb.preview_size == (
                W,
                H,
            ), f"Yolo network expects input size ({W}, {H}) but camera gives ({self.camRgb.preview_size[0]}, {self.camRgb.preview_size[1]})"
            self.camRgb.cam.setInterleaved(False)

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
            self.lv_rgb = LiveView(self.pm, self, "rgb", preview_modifier)
            self.lv_nn = LiveView(self.pm, self, "out", lambda d: d)

        # Linking
        self.camRgb.linkTo(self, "preview", "input")

    @property
    def inputs(self) -> list[str]:
        return self._inputs.keys()

    def get_input(self, name: str):
        assert name in self._inputs
        return self._inputs[name]

    @property
    def outputs(self) -> list[str]:
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

    def draw_frame(
        self, name: str, frame=None, new_detections=True, color=(255, 0, 0), show=True
    ):
        """Displays the video frame and draws detection results

        Args:
            name (str): Name of the cv2 window.
            frame (_type_, optional): The frame of the video feed. Defaults to None (we get the frame internally).
            new_detections (bool, optional): Retrieve brand-new detections from the device or used cached ones. Defaults to True.
            color (Tuple[int]): bgr color for drawing frames
            show (boolean): whether to show in a cv2 frame
        Returns:
            np.arry: frame with drawn bboxes, scores and labels
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
        if show:
            cv2.imshow(name, frame)

        return frame


class YoloSpatialDetector(ObjectDetector):
    """Object detector using yolo model"""

    def __init__(
        self,
        pm: PipelineManager,
        config: Path,
        blob: Path,
        camRgb: ColorCamera,
        stereo: StereoDepth,
        provide_rgb=True,
        spatialCalculationAlgorithm=dai.SpatialLocationCalculatorAlgorithm.AVERAGE,
    ):
        super().__init__(pm)

        self.provide_rgb = provide_rgb
        self.camRgb = camRgb
        self.stereo = stereo
        self.spatialCalculationAlgorithm = spatialCalculationAlgorithm

        assert config.exists()
        assert blob.exists()

        with open(config, encoding="utf-8") as config_input:
            self.yolo_config = json.load(config_input)

        self.__create_detection_network(blob)

        # sync outputs
        # syncNN = True

        # Cache for detections and frame
        self.last_detections = None
        self.last_frame = None

    def __create_detection_network(self, blob):
        nnConfig = self.yolo_config.get("nn_config", {})

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
        nnMappings = self.yolo_config.get("mappings", {})
        self.labels = nnMappings.get("labels", {})

        spatialDetectionNetwork = self.pm.pipeline.create(
            dai.node.YoloSpatialDetectionNetwork
        )

        if isinstance(self.camRgb, ColorCamera):
            assert self.camRgb.preview_size == (
                W,
                H,
            ), f"Yolo network expects input size ({W}, {H}) but camera gives ({self.camRgb.preview_size[0]}, {self.camRgb.preview_size[1]})"
            self.camRgb.cam.setInterleaved(False)

        # Network specific settings
        spatialDetectionNetwork.setConfidenceThreshold(confidenceThreshold)
        spatialDetectionNetwork.setNumClasses(classes)
        spatialDetectionNetwork.setCoordinateSize(coordinates)
        spatialDetectionNetwork.setAnchors(anchors)
        spatialDetectionNetwork.setAnchorMasks(anchorMasks)
        spatialDetectionNetwork.setIouThreshold(iouThreshold)
        spatialDetectionNetwork.setBlobPath(blob)
        spatialDetectionNetwork.setNumInferenceThreads(2)
        spatialDetectionNetwork.input.setBlocking(False)

        if self.spatialCalculationAlgorithm:
            spatialDetectionNetwork.setSpatialCalculationAlgorithm(
                self.spatialCalculationAlgorithm
            )

        # define inputs and outputs
        self._inputs = {
            "input": spatialDetectionNetwork.input,
            "inputDepth": spatialDetectionNetwork.inputDepth,
        }
        self._outputs = {
            "rgb": spatialDetectionNetwork.passthrough,
            "passthrough": spatialDetectionNetwork.passthrough,
            "out": spatialDetectionNetwork.out,
            "passthroughDepth": spatialDetectionNetwork.passthroughDepth,
        }

        if self.provide_rgb:
            self.lv_rgb = LiveView(self.pm, self, "rgb", preview_modifier)
            self.lv_nn = LiveView(self.pm, self, "out", lambda d: d)

        # Linking
        self.camRgb.linkTo(self, "preview", "input")
        self.stereo.linkTo(self, "depth", "inputDepth")

    @property
    def inputs(self) -> list[str]:
        return self._inputs.keys()

    def get_input(self, name: str):
        assert name in self._inputs
        return self._inputs[name]

    @property
    def outputs(self) -> list[str]:
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

    def displayFrame(
        self, name: str, frame=None, new_detections=True, color=(255, 0, 0)
    ):
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
        height = frame.shape[0]
        width = frame.shape[1]
        for detection in detections:
            # Denormalize bounding box
            x1 = int(detection.xmin * width)
            x2 = int(detection.xmax * width)
            y1 = int(detection.ymin * height)
            y2 = int(detection.ymax * height)
            try:
                label = self.labels[detection.label]
            except KeyError:
                label = detection.label
            cv2.putText(
                frame,
                str(label),
                (x1 + 10, y1 + 20),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                color,
            )
            cv2.putText(
                frame,
                f"{detection.confidence*100:.2f}",
                (x1 + 10, y1 + 35),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                color,
            )
            cv2.putText(
                frame,
                f"X: {int(detection.spatialCoordinates.x)} mm",
                (x1 + 10, y1 + 50),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                color,
            )
            cv2.putText(
                frame,
                f"Y: {int(detection.spatialCoordinates.y)} mm",
                (x1 + 10, y1 + 65),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                color,
            )
            cv2.putText(
                frame,
                f"Z: {int(detection.spatialCoordinates.z)} mm",
                (x1 + 10, y1 + 80),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                color,
            )

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

        # Show the frame
        cv2.imshow(name, frame)


class SpatialDetector(ObjectDetector):
    """Spatial detector node"""


class ObjectTrackerConfig:
    """Configuration for object tracking"""

    def __init__(
        self,
        labels: list[str | int] = None,
        tracker_type: dai.TrackerType = dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM,
        id_assignment_policy=dai.TrackerIdAssignmentPolicy.SMALLEST_ID,
    ) -> None:
        """Create object tracker config

        Args:
            labels (List[str  |  int], optional): List of labes that are tracked. Both label index (int) and label name (str) are allowed. Defaults to None.
            type (dai.TrackerType, optional): method to track detected objects. Defaults to dai.TrackerType.ZERO_TERM_COLOR_HISTOGRAM.
            id_assignment_policy (_type_, optional): assignment policy for new objects. Defaults to dai.TrackerIdAssignmentPolicy.SMALLEST_ID.
        """
        self.labels = labels
        self.tracker_type = tracker_type
        self.id_assignment_policy = id_assignment_policy


class ObjectTracker(Node, InputOutput):
    """Object tracking node"""

    def __init__(
        self,
        pm: PipelineManager,
        od: ObjectDetector,
        cam: ColorCamera = None,
        use_full_camera_view=False,
        config=ObjectTrackerConfig(),
    ):
        """Create object tracker node

        Args:
            pm (PipelineManager): the global pipeline manager.
            od (ObjectDetector): object detector node.
            cam (ColorCamera, optional): camera node can be used for visualization on higher resolution image stream. Not needed for tracking. Defaults to None.
            use_full_camera_view (bool, optional): use the camera to visualize tracking on full view. Defaults to False.
            config (_type_, optional): static configuration for tracking. Defaults to ObjectTrackerConfig().

        Raises:
            ValueError: _description_
        """
        super().__init__(pm)

        self.od = od

        # create depthai object tracker
        self.objectTracker = objectTracker = pm.pipeline.create(dai.node.ObjectTracker)

        if config.labels:
            # convert possible string labels to integers
            integer_labels = list(
                map(
                    lambda l: l if isinstance(l, int) else self.od.labels.index(l),
                    config.labels,
                )
            )
            # set the tracker to track only some labels
            objectTracker.setDetectionLabelsToTrack(integer_labels)

        # possible tracking types: ZERO_TERM_COLOR_HISTOGRAM, ZERO_TERM_IMAGELESS, SHORT_TERM_IMAGELESS, SHORT_TERM_KCF
        objectTracker.setTrackerType(config.tracker_type)
        # take the smallest ID when new object is tracked, possible options: SMALLEST_ID, UNIQUE_ID
        objectTracker.setTrackerIdAssignmentPolicy(config.id_assignment_policy)

        # define intputs and outputs
        self._inputs = {
            "inputTrackerFrame": objectTracker.inputTrackerFrame,
            "inputDetectionFrame": objectTracker.inputDetectionFrame,
            "inputDetections": objectTracker.inputDetections,
        }
        self._outputs = {
            "passthroughTrackerFrame": objectTracker.passthroughTrackerFrame,
            "out": objectTracker.out,
        }

        # view of tracker frame
        self.lv_tracker_frame = LiveView(
            pm, self, "passthroughTrackerFrame", lambda d: d
        )

        # link vidoe frame from object detector to me
        if use_full_camera_view and cam:
            # get video stream from camera default
            cam.linkTo(self, cam.get_default_output(), "inputTrackerFrame")
        elif use_full_camera_view:
            # forgot to specify camera
            raise ValueError("You need to specify a camera for that")
        else:
            # get the video stream from object detector passthrough
            od.linkTo(self, "passthrough", "inputTrackerFrame")
        # link detection frame from object detector to me
        od.linkTo(self, "passthrough", "inputDetectionFrame")

        # link detections to tracker input
        od.linkTo(self, "out", "inputDetections")

        # output for object tracker
        self.lv_tracker_output = LiveView(pm, self, "out", lambda d: d)

    @property
    def inputs(self):
        return self._inputs.keys()

    def get_input(self, name: str):
        return self._inputs[name]

    @property
    def outputs(self):
        return self._outputs.keys()

    def get_output(self, name: str):
        return self._outputs[name]

    def get_frame(self):
        """Read frame from device"""
        return self.lv_tracker_frame.get().getCvFrame()

    def get_tracklets(self):
        """Read tracklets from device"""
        return self.lv_tracker_output.get().tracklets

    def draw_frame(self, frame=None, tracklets=None, show=True):
        """Draw the tracklets on frame

        Args:
            frame (_type_, optional): frame to use for visualization. If none we try to grab it from the object detector. Defaults to None.
            tracklets (_type_, optional): tracklets to visualize. If not specified we get it from the tracker node. Defaults to None.
        """
        if frame is None:
            # get frame
            frame = self.lv_tracker_frame.get().getCvFrame()
        if tracklets is None:
            # get tracklet
            tracklets = self.lv_tracker_output.get().tracklets

        color = (255, 0, 0)
        for t in tracklets:
            # loop over all tracklets
            roi = t.roi.denormalize(frame.shape[1], frame.shape[0])

            # copute the bounding box coordinates
            x1 = int(roi.topLeft().x)
            y1 = int(roi.topLeft().y)
            x2 = int(roi.bottomRight().x)
            y2 = int(roi.bottomRight().y)

            # obtain the label (at best in string coordinates)
            try:
                label = self.od.labels[t.label]
            except ValueError:
                label = t.label

            # draw onto frame
            cv2.putText(
                frame,
                str(label),
                (x1 + 10, y1 + 20),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                255,
            )
            cv2.putText(
                frame,
                f"ID: {[t.id]}",
                (x1 + 10, y1 + 35),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                255,
            )
            cv2.putText(
                frame,
                t.status.name,
                (x1 + 10, y1 + 50),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.5,
                255,
            )
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

        if show:
            # show frame
            cv2.imshow("tracker", frame)

        return frame


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
        disparity_size: tuple[int, int] = None,
        confidence_threshold: int = 255,
    ):

        self.rectify_edge_fill_color = rectify_edge_fill_color
        self.left_right_check = left_right_check
        self.subpixel = subpixel
        self.extended_disparity = extended_disparity
        self.default_profile_preset = default_profile_preset
        self.median_filter = median_filter
        self.depth_align = depth_align

        self.input_resolution = input_resolution

        self.disparity_size = disparity_size

        self.confidence_threshold = confidence_threshold

        assert not (
            subpixel and extended_disparity
        ), "Subpixel and ExtendedDisparity are not supported at the same time"


class StereoDepth(Node, InputOutput):
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

        if config.depth_align:
            self.stereo.setDepthAlign(config.depth_align)

        if config.input_resolution:
            self.stereo.setInputResolution(*config.input_resolution)

        if config.disparity_size:
            self.stereo.setOutputSize(*config.disparity_size)

        if not config.confidence_threshold is None:
            self.stereo.setConfidenceThreshold(config.confidence_threshold)

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
