"""Main module."""

import logging

import depthai as dai


class PipelineManager:
    """Hihg-level pipeline manager"""

    xstream_names = set()

    def __init__(self):
        self.pipeline = dai.Pipeline()
        self.device = None

        self.nodes = []

    def __enter__(self):
        self.device = self.createDevice()
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # exit OAK-D device
        self.device.__exit__(exc_type, exc_value, exc_traceback)
        self.device = None

        # close all nodes
        for node in self.nodes:
            node.deactivate()

    def createDevice(self):
        self.device = dai.Device(self.pipeline)

        for node in self.nodes:
            node.activate(self.device)
        return self.device

    def add_xstream_name(self, name: str) -> str:
        """Allows to safely add an xlink stream name. If the name already exists, a new unique will be generated.

        Args:
            name (str): the preferred xstream name.

        Returns:
            str: unique xstream name
        """
        assert name

        if name in self.xstream_names:
            # generate new name
            temp_name = str(name)

            index = 0
            while temp_name in self.xstream_names:
                temp_name = f"{name}_{index}"
                index += 1

            logging.info("Rename requested stream: %s -> %s", name, temp_name)

            name = temp_name

        self.xstream_names.add(name)

        return name
