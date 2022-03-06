"""
Example for aligned rgb and depth information.

Based on official example: https://github.com/luxonis/depthai-python/blob/main/examples/StereoDepth/rgb_depth_aligned.py

"""

from fractions import Fraction
from functools import partial

import cv2
import depthai as dai
import numpy as np

from depthai_lightning import PipelineManager
from depthai_lightning.nodes import (
    ColorCamera,
    LiveView,
    MonoCamera,
    StereoConfig,
    StereoDepth,
)


def disparity_modifier(inFrame, stereo_node):
    """Extract 8-bit disparity frame from depthai package.

    Args:
        inFrame (_type_): depthai pacakge
        stereo_node (_type_): the stereo node

    Returns:
        np.array[np.uint8]: colored disparity image
    """
    maxDisp = stereo_node.stereo.initialConfig.getMaxDisparity()
    frame = inFrame.getCvFrame()
    disp = (frame * (255.0 / maxDisp)).astype(np.uint8)
    disp = cv2.applyColorMap(disp, cv2.COLORMAP_JET)

    return disp


class Weights:
    """Class for storing rgb/depth blending weights"""

    def __init__(self):
        self.rgbWeight = 0.4

    @property
    def depthWeight(self):
        return 1 - self.rgbWeight

    def updateBlendWeights(self, percent_rgb):
        """
        Update the rgb and depth weights used to blend depth/rgb image

        @param[in] percent_rgb The rgb weight expressed as a percentage (0..100)
        """
        self.rgbWeight = float(percent_rgb) / 100.0


weights = Weights()

# The disparity is computed at this resolution, then upscaled to RGB resolution
monoResolution = 720
rgbResolution = 1080

# scale applied to rgb to reduce to mono resolution
# in default config this should be: (4,9)
isp_scale = Fraction(monoResolution, rgbResolution)

if __name__ == "__main__":
    pm = PipelineManager()

    rgb_cam = ColorCamera(
        pm,
        resolution=str(rgbResolution),
        isp_scale=(isp_scale.numerator, isp_scale.denominator),
    )
    left = MonoCamera(pm, dai.CameraBoardSocket.LEFT, monoResolution)
    right = MonoCamera(pm, dai.CameraBoardSocket.RIGHT, monoResolution)

    stereoConfig = StereoConfig(
        left_right_check=True, subpixel=True, depth_align=dai.CameraBoardSocket.RGB
    )

    stereo = StereoDepth(pm, left, right, stereoConfig)

    cam_lv = LiveView(pm, rgb_cam, "isp")
    # left_lv = LiveView(pm, left)
    # right_lv = LiveView(pm, right)
    disp_lv = LiveView(
        pm,
        stereo,
        "disparity",
        data_modifier=partial(disparity_modifier, stereo=stereo),
    )

    with pm.createDevice() as device:

        # Configure windows; trackbar adjusts blending ratio of rgb/depth
        rgbWindowName = "rgb"
        dispWindowName = "disparity"
        blendedWindowName = "rgb-depth"
        cv2.namedWindow(rgbWindowName)
        cv2.namedWindow(dispWindowName)
        cv2.namedWindow(blendedWindowName)
        cv2.createTrackbar(
            "RGB Weight %",
            blendedWindowName,
            int(weights.rgbWeight * 100),
            100,
            weights.updateBlendWeights,
        )

        while True:
            rgb_frame = cam_lv.get()
            disp_frame = disp_lv.get()

            # show rgb and disparity
            cv2.imshow(rgbWindowName, rgb_frame)
            cv2.imshow(dispWindowName, disp_frame)

            # blend both to show combination
            blended = cv2.addWeighted(
                rgb_frame, weights.rgbWeight, disp_frame, weights.depthWeight, 0
            )
            cv2.imshow(blendedWindowName, blended)

            if cv2.waitKey(1) == ord("q"):
                break
