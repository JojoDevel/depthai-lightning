"""
Simple example for visualizing depth and camera preview
"""

import cv2
import depthai as dai

from depthai_lightning.depthai_lightning import PipelineManager
from depthai_lightning.nodes import ColorCamera, LiveView, MonoCamera, StereoDepth


def depth_modifier(inFrame):
    return inFrame.getCvFrame()


if __name__ == "__main__":

    pm = PipelineManager()

    # create left/right cameras
    left = MonoCamera(pm, dai.CameraBoardSocket.LEFT, 480)
    right = MonoCamera(pm, dai.CameraBoardSocket.RIGHT, 480)

    # create stereo depth with default parameters
    depth = StereoDepth(pm, left, right)

    # also create color camera
    color = ColorCamera(pm)

    # live view for depth and camera
    lv = LiveView(pm, depth, "depth", depth_modifier)
    cam_lv = LiveView(pm, color, "preview")

    with pm.createDevice() as device:
        while True:
            # show live views
            lv.show("depth")
            cam_lv.show("camera")

            if cv2.waitKey(1) == ord("q"):
                break
