"""Example for object detection and tracking using Yolo
"""

import argparse
from pathlib import Path

import cv2

from depthai_lightning import PipelineManager
from depthai_lightning.nodes import ColorCamera
from depthai_lightning.nodes.processing import (  # ObjectTrackerConfig,
    ObjectTracker,
    YoloDetector,
)
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
        "-m", "--model", type=str, help="Path to the myriad X ready blob file."
    )
    args = parser.parse_args()

    # create pipeline manager
    pm = PipelineManager()

    # create color camera with preview size for yolo
    cam = ColorCamera(pm, preview_size=(416, 416), fps=35)

    # create the yolo object detector
    yd = YoloDetector(pm, Path(args.config), Path(args.model), cam)

    # Three examples for trackers

    # simple tracker
    tracker = ObjectTracker(pm, yd)

    # only track persons
    # tracker = ObjectTracker(pm, yd, config=ObjectTrackerConfig(labels=["person"], id_assignment_policy=dai.TrackerIdAssignmentPolicy.UNIQUE_ID))

    # show tracks on full view (it is not tracked full view)
    # tracker = ObjectTracker(pm, yd, cam=cam, use_full_camera_view=True)

    counter = FPSCounter()

    # dispatch pipeline
    with pm:
        # start fps counter
        counter.start()
        while True:
            # get frame and tracklets
            frame = tracker.get_frame()  # tracker.lv_tracker_frame.get().getCvFrame()
            tracklets = (
                tracker.get_tracklets()
            )  # tracker.lv_tracker_output.get().tracklets

            # count and draw fps to frame
            counter.tick()
            counter.publish(frame)

            # draw tracklets onto frame
            tracker.draw_frame(frame, tracklets)

            if cv2.waitKey(1) == ord("q"):
                break