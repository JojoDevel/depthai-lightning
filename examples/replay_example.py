"""
Example for replaying recorded streams and depth computation
"""

import argparse
from functools import partial

import cv2
import numpy as np

from depthai_lightning.depthai_lightning import PipelineManager
from depthai_lightning.nodes import (
    LiveView,
    Replay,
    StereoConfig,
    StereoDepth,
    preview_modifier,
)


def disp_mode(inData, disparity_multiplier: float):
    """Apply colormap to disparity frame

    Args:
        inData (_type_): package data from xlink
        disparity_multiplier (float): multiplier to scale disparity to [0...255]

    Returns:
        np.arry: depth frame with applied JET colormap
    """
    frame = preview_modifier(inData)

    frame = (frame * disparity_multiplier).astype(np.uint8)

    return cv2.applyColorMap(frame, cv2.COLORMAP_JET)


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser("Replay Example")
    parser.add_argument(
        "path", type=str, help="Path to the folder containing the stream recordings"
    )
    args = parser.parse_args()

    # create pipeline manager
    pm = PipelineManager()

    # create replay node to read streams from files
    rp = Replay(pm, args.path)

    # create stereo node based on replay inputs
    stereo = StereoDepth(
        pm,
        rp.left,
        rp.right,
        StereoConfig(input_resolution=rp.size["left"], subpixel=True),
    )

    # compute the disparity multiplier (to go to range [0...255])
    disparityMultiplier = 255 / stereo.stereo.initialConfig.getMaxDisparity()

    # create individual streams
    mono_left = LiveView(pm, stereo, "rectifiedLeft", preview_modifier)
    mono_right = LiveView(pm, stereo, "rectifiedRight", preview_modifier)
    disparity = LiveView(
        pm,
        stereo,
        "disparity",
        partial(disp_mode, disparity_multiplier=disparityMultiplier),
    )
    depth = LiveView(pm, stereo, "depth", preview_modifier)

    # create device with pipeline
    with pm.createDevice() as device:
        while True:
            # send frames to device
            rp.send_frames()

            # show streams
            mono_left.show("left")
            mono_right.show("right")
            disparity.show("disp")
            depth.show("depth")

            if cv2.waitKey(1) == ord("q"):
                break
