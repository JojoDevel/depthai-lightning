"""Main module."""

import depthai as dai


class PipelineManager:
    """Hihg-level pipeline manager"""

    def __init__(self):
        self.pipeline = dai.Pipeline()
        self.device = None

        self.nodes = []

    def createDevice(self):
        self.device = dai.Device(self.pipeline)

        for node in self.nodes:
            node.activate(self.device)
        return self.device
