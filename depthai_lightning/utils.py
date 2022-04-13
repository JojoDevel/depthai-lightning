"""
Utilities module
"""
import time

import cv2
import numpy as np


class FPSCounter:
    """Counts frame events to compute fps (using median)"""

    def __init__(self, max_size=30, method=np.median):
        """_summary_

        Args:
            max_size (int, optional): Maximum size of the cached durations. Defaults to 30.
            method (_type_, optional): method to compute average/median from list of cache durations. Defaults to np.median.
        """
        self.start_time = None
        self.last_publish = None
        self.durations = []
        self.max_size = max_size
        self.method = method

    def start(self):
        """Start time measuring"""
        self.start_time = time.time()

    def tick(self):
        """Take note of a frame event"""
        now = time.time()
        self.durations.append(now - self.start_time)
        self.start_time = now
        self.durations = self.durations[-self.max_size :]

    @property
    def fps(self) -> float:
        """Compute current fps

        Returns:
            float: fps
        """
        return 1.0 / self.method(self.durations)

    def reset(self):
        """Reste the counter"""
        self.start_time = time.time()
        self.durations = []

    def publish(self, frame=None, every_n_seconds=5):
        """Publish fps

        Args:
            frame (np.arry, optional): writes fps onto cv2 frame
            every_n_seconds (int, optional): publishes every n seconds. Defaults to 5.
        """
        if frame is not None:
            color = (255, 255, 255)
            cv2.putText(
                frame,
                f"NN fps: {self.fps:.2f}",
                (2, frame.shape[0] - 4),
                cv2.FONT_HERSHEY_TRIPLEX,
                0.4,
                color,
            )
        if self.last_publish is None or (
            time.time() - self.last_publish > every_n_seconds
        ):
            print(f"FPS: {self.fps}")
            self.last_publish = time.time()
