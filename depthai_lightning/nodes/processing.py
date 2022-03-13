""" High-level data processing nodes"""

import depthai as dai

from depthai_lightning.depthai_lightning import PipelineManager
from depthai_lightning.nodes.modifiers import preview_modifier
from .input import MonoCamera

from .base import Node


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

    def get_data_modifier(self, output_name):
        assert output_name in self.outputs

        return preview_modifier


class ImageCrop(Node):
    """image cropping node"""


class PythonNode(Node):
    """node for custom python node on-device"""
