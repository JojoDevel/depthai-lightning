"""Simple example for camera usage
"""

import cv2

from depthai_lightning import PipelineManager
from depthai_lightning.nodes import CameraColor

if __name__ == "__main__":

    pm = PipelineManager()

    # create high-level nodes
    cam = CameraColor(pm)
    liveView = cam.liveView(stream="preview")  # or raw, isp

    with pm.createDevice() as device:
        while True:
            # show video stream
            liveView.show()

            if cv2.waitKey(1) == ord("q"):
                break
