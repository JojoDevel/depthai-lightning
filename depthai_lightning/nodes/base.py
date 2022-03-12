"""Base classes for high-level nodes"""

from typing import List

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
