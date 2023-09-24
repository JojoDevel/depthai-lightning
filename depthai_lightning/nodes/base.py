"""Base classes for high-level nodes"""

from typing import Any, List

import depthai as dai

from depthai_lightning import PipelineManager


class InputOutput:
    """Provides abstract class for input and output linking

    Raises:
        NotImplementedError: input/output endpoints have to be implemented by subclasses.
        ValueError: when linking fails due to wrong input/output names
    """

    @property
    def inputs(self) -> List[str]:
        """List all possible input names.

        These input names can be used to fill their input with the output of another node.

        Returns:
            List[str]: list of all input names.
        """
        raise NotImplementedError()

    @property
    def outputs(self) -> List[str]:
        """List all possible output names.

        These output names can be used to fill inputs of other nodes

        Returns:
            List[str]: list of all output names
        """
        raise NotImplementedError()

    def get_input(self, name: str):
        """get the depthai object of an input name.

        Args:
            name (str): input name

        Raises:
            NotImplementedError: Please implement this abstract method
        """
        raise NotImplementedError()

    def get_output(self, name: str):
        """get the depthai object of an output name.

        Args:
            name (str): output name

        Raises:
            NotImplementedError: Please implement this abstract method
        """
        raise NotImplementedError()

    def __getitem__(self, name: str):
        """get input or output name

        Args:
            name (str): input or output name

        Raises:
            ValueError: if name does not match any input or output name

        Returns:
            _type_: _description_
        """
        if name in self.inputs:
            return self.get_input(name)

        if name in self.outputs:
            return self.get_output(name)

        raise ValueError(f"{name} is neither input or output of this node.")

    def linkTo(self, target: "Node", my_output: str = None, target_input: str = None):
        """Links an output of this node to an input of another node

        Args:
            target (Node): target node
            my_output (str, optional): this nodes output name. Defaults to None (using default output).
            target_input (str, optional): the target input name. Defaults to None (using default input).

        Raises:
            ValueError: when default input/output is not available (multiple possibilities) and no specifc value has been specified.
        """
        inputs = target.inputs
        outputs = self.outputs

        selected_output = None
        selected_input = None
        if my_output:
            selected_output = self.get_output(my_output)
        else:
            if len(outputs) == 1:
                # choose default output
                selected_output = self.get_output(outputs[0])
            else:
                raise ValueError(
                    f"Multiple outputs available {outputs} but none is specified!"
                )

        if target_input:
            selected_input = target.get_input(target_input)
        else:
            if len(inputs) == 1:
                # choose default input
                selected_input = target.get_input(inputs[0])
            else:
                raise ValueError(
                    f"Multiple outputs available {outputs} but none is specified!"
                )

        assert selected_output and selected_input

        # link my output to target's input
        selected_output.link(selected_input)


class Node:
    """Base class for high-level nodes"""

    def __init__(self, pm: PipelineManager):
        self.pm = pm

        self.pm.nodes.append(self)

    def activate(self, device: dai.Device):
        pass

    def deactivate(self):
        pass


class LazyXNode(Node):
    """Node for lazy xnode creation. XLinkNode is only created on direct request and usage for other node."""

    def __init__(self, pm: Node, xnode_creator, host_node_creator=None):
        """_summary_

        Args:
            pm (Node): global pipeline manager
            xnode_creator (_type_): function to create xnode on devcie
            host_node_creator (_type_, optional): function to create similar host node. Defaults to None.
        """
        super().__init__(pm)
        self.node = None
        self.xnode_creator = xnode_creator
        self.host_node_creator = host_node_creator

        self.x_link_pipeline_created = False

    def activate(self, device: dai.Device):
        self.x_link_pipeline_created = True

    def __create(self):
        if self.node is None:
            if self.x_link_pipeline_created is False:
                # create node
                self.node = self.xnode_creator()
            else:
                # node was not linke before pipeline creation
                if self.host_node_creator is not None:
                    self.node = self.host_node_creator()
                else:
                    raise ValueError(
                        "Your Node has not been linked in the pipeline before the device creation! And you have not added a creator for the host node"
                    )

        return self.node

    def __getattribute__(self, __name: str) -> Any:
        try:
            return super().__getattribute__(__name)
        except AttributeError:
            # create the node on the device only on request
            node = self.__create()
            return getattr(node, __name)
