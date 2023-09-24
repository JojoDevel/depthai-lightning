"""Yolov5 object detection example for recordings
"""

import argparse
from pathlib import Path

import cv2

from depthai_lightning import PipelineManager
from depthai_lightning.nodes.input import Replay
from depthai_lightning.nodes.processing import YoloDetector
from depthai_lightning.utils import FPSCounter

if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default="yolov5.json",
        help="Path to yolo object detector network.",
    )
    parser.add_argument(
        "recording_path", type=str, help="Path to the recording folder."
    )
    parser.add_argument(
        "-m", "--model", type=str, help="Path to the myriad X ready blob file."
    )
    args = parser.parse_args()

    # create pipeline manager
    pm = PipelineManager()

    # create replay node to read streams from files
    rp = Replay(
        pm, args.recording_path, streams=["color"], color_size=(416, 416), keep_ar=False
    )

    # create the yolo object detector
    yd = YoloDetector(pm, Path(args.config), Path(args.model), rp.color)

    # framerate counter
    fpsC = FPSCounter()

    # dispatch pipeline
    with pm:
        fpsC.start()
        while True:
            # send recorded frames to device
            rp.send_frames()

            # get frame from object detector
            frame = yd.get_frame()

            # add fps info
            fpsC.tick()
            fpsC.publish(frame=frame)

            # show rgb frame with detections
            yd.displayFrame("detections", frame=frame)

            if cv2.waitKey(1) == ord("q"):
                break
