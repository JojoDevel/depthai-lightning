"""Simple example for camera usage
"""

import cv2

from depthai_lightning import PipelineManager
from depthai_lightning.nodes import ColorCamera, LiveView

if __name__ == "__main__":

    pm = PipelineManager()

    # create high-level nodes
    cam = ColorCamera(pm)
    liveView = LiveView(pm, cam, "isp")  # or raw, isp

    with pm.createDevice() as device:
        while True:
            # show video stream
            liveView.show()

            if cv2.waitKey(1) == ord("q"):
                break
