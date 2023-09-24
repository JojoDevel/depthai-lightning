"""Simple example for camera usage
"""

from pathlib import Path

import cv2

from depthai_lightning import PipelineManager
from depthai_lightning.nodes import ColorCamera, EncodingConfig, LiveView, VideoEncoder

if __name__ == "__main__":

    pm = PipelineManager()

    # create high-level nodes

    # color camera
    cam = ColorCamera(pm)
    # live view for video
    liveView = LiveView(pm, cam, "video")  # or raw, preview
    # encoder for video
    enc = VideoEncoder(pm, Path("record"), EncodingConfig.LOW, cam, True)

    # create the pipeline
    with pm:
        while True:
            # show video stream
            liveView.show()
            # encode video stream
            enc.write()

            if cv2.waitKey(1) == ord("q"):
                break
