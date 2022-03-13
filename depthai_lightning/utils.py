"""
Utilities module
"""
import time


class FPSCounter:
    """Counts frame events to compute fps"""

    def __init__(self):
        self.start_time = None
        self.last_publish = None
        self.counter = 0

    def start(self):
        """Start time measuring"""
        self.start_time = time.time()

    def tick(self):
        """Take note of a frame event"""
        self.counter += 1

    @property
    def fps(self) -> float:
        """Compute current fps

        Returns:
            float: fps
        """
        duration = time.time() - self.start_time
        return self.counter / duration

    def reset(self):
        """Reste the counter"""
        self.start_time = time.time()
        self.counter = 0

    def publish(self, every_n_seconds=5, reset_on_publish=True):
        """Publish fps

        Args:
            every_n_seconds (int, optional): publishes every n seconds. Defaults to 5.
            reset_on_publish (bool, optional): resets the counter after publishing. Defaults to True.
        """
        if self.last_publish is None or (
            time.time() - self.last_publish > every_n_seconds
        ):
            print(f"FPS: {self.fps}")
            self.last_publish = time.time()
            if reset_on_publish:
                self.reset()
