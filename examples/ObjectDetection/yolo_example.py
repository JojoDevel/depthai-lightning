"""Simple example for camera usage
"""

import argparse
from pathlib import Path

import cv2

from depthai_lightning import PipelineManager
from depthai_lightning.nodes import ColorCamera
from depthai_lightning.nodes.processing import YoloDetector

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
        "-m", "--model", type=str, help="Path to the myriad X ready blob file."
    )
    args = parser.parse_args()

    # create pipeline manager
    pm = PipelineManager()

    # create color camera with preview size for yolo
    cam = ColorCamera(pm, preview_size=(416, 416))

    # create the yolo object detector
    yd = YoloDetector(pm, Path(args.config), Path(args.model), cam)

    # dispatch pipeline
    with pm:
        while True:
            # show rgb frame with detections
            yd.displayFrame("detections")

            if cv2.waitKey(1) == ord("q"):
                break
