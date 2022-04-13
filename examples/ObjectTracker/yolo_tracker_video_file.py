"""Example for object detection and tracking using Yolo on video data.
"""

import argparse
from pathlib import Path

import cv2
import depthai as dai

from depthai_lightning import PipelineManager
from depthai_lightning.nodes.input import Video
from depthai_lightning.nodes.processing import (
    ObjectTracker,
    ObjectTrackerConfig,
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
    parser.add_argument(
        "recording_file", type=str, help="Path to the folder containing the recording."
    )
    args = parser.parse_args()

    # create pipeline manager
    pm = PipelineManager()

    # create color camera with preview size for yolo
    videoInput = Video(
        pm, args.recording_file, mode="color", resize_size=(416, 416), keep_ar=False
    )
    videoInputOrig = Video(pm, args.recording_file, mode="color")

    # create the yolo object detector
    yd = YoloDetector(pm, Path(args.config), Path(args.model), videoInput.color)

    # Three examples for trackers

    # simple tracker
    tracker = ObjectTracker(
        pm,
        yd,
        config=ObjectTrackerConfig(
            id_assignment_policy=dai.TrackerIdAssignmentPolicy.UNIQUE_ID
        ),
    )

    # only track persons
    # tracker = ObjectTracker(pm, yd, config=ObjectTrackerConfig(labels=["person"], id_assignment_policy=dai.TrackerIdAssignmentPolicy.UNIQUE_ID))

    # show tracks on full view (it is not tracked full view)
    # tracker = ObjectTracker(pm, yd, cam=cam, use_full_camera_view=True)

    counter = FPSCounter()

    # dispatch pipeline
    with pm:
        # start fps counter
        counter.start()

        # pre-fil the pipeline (dramatically increases performance (~2x) because no longer data transfer bound)
        videoInput.send_frame()
        videoInput.send_frame()
        while True:
            videoInput.send_frame()
            # get frame and tracklets
            frame = videoInputOrig.read_frame()  # tracker.get_frame()
            tracklets = tracker.get_tracklets()

            # count and draw fps to frame
            counter.tick()
            counter.publish(frame, min_ticks=20)

            # draw tracklets onto frame
            frame = tracker.draw_frame(frame, tracklets, show=False)

            resize_frame = cv2.resize(frame, videoInput.size)
            cv2.imshow("tracking", resize_frame)

            if cv2.waitKey(1) == ord("q"):
                break
