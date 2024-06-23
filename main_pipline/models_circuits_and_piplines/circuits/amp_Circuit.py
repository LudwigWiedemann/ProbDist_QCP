import math
import sys
from functools import partial

import numpy as np
import pennylane as qml


class AmpCircuit:
    def __init__(self):
        super().__init__()

    def run(self, inputs, weights_0, weights_1, weights_2):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_weights(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_wires(self):
        raise NotImplementedError("This method should be implemented by subclasses.")


class base_Amp_Circuit(AmpCircuit):
    def __init__(self, config):
        super().__init__()
        self.n_wires = int(math.log2(config['time_steps']))
        self.weight_shapes = {"weights_RY": self.n_wires, "weights_RX": self.n_wires, "weights_RZ": self.n_wires}

    def run(self):
        training_device = qml.device("default.qubit", wires=self.n_wires)

        @partial(qml.batch_input, argnum=0)
        @qml.qnode(training_device, interface='tf')
        def example_complex_circuit(inputs, weights_RY, weights_RX, weights_RZ):
            qml.AmplitudeEmbedding(features=inputs, wires=range(self.n_wires), normalize=True)
            qml.broadcast(qml.RY, wires=range(self.n_wires), pattern="single", parameters=weights_RY)
            qml.broadcast(qml.RX, wires=range(self.n_wires), pattern="single", parameters=weights_RX)
            qml.broadcast(qml.RZ, wires=range(self.n_wires), pattern="single", parameters=weights_RZ)
            qml.broadcast(qml.CNOT, wires=range(self.n_wires), pattern="all_to_all")
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

        return example_complex_circuit

    def get_weights(self):
        return self.weight_shapes

    def get_wires(self):
        return self.n_wires


class layered_Amp_Circuit(AmpCircuit):
    def __init__(self, config):
        super().__init__()
        self.n_wires = int(math.log2(config['time_steps']))
        self.weight_shapes = {"w_RY": self.n_wires,
                              "w_RX": self.n_wires,
                              "w_RZ": self.n_wires,
                              "w_RY_2": self.n_wires,
                              "w_RX_2": self.n_wires,
                              "w_RZ_2": self.n_wires}

    def run(self):
        training_device = qml.device("default.qubit", wires=self.n_wires)

        @partial(qml.batch_input, argnum=0)
        @qml.qnode(training_device, interface='tf')
        def circuit(inputs, w_RY, w_RX, w_RZ, w_RY_2, w_RX_2, w_RZ_2):
            qml.broadcast(qml.CNOT, wires=range(self.n_wires), pattern="all_to_all")

            qml.AmplitudeEmbedding(features=inputs, wires=range(self.n_wires), normalize=True)
            qml.broadcast(qml.RY, wires=range(self.n_wires), pattern="single", parameters=w_RY)
            qml.broadcast(qml.RX, wires=range(self.n_wires), pattern="single", parameters=w_RX)
            qml.broadcast(qml.RZ, wires=range(self.n_wires), pattern="single", parameters=w_RZ)

            qml.broadcast(qml.CNOT, wires=range(self.n_wires), pattern="all_to_all")

            qml.broadcast(qml.RZ, wires=range(self.n_wires), pattern="single", parameters=w_RZ_2)
            qml.broadcast(qml.RX, wires=range(self.n_wires), pattern="single", parameters=w_RX_2)
            qml.broadcast(qml.RY, wires=range(self.n_wires), pattern="single", parameters=w_RY_2)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

        return circuit

    def get_weights(self):
        return self.weight_shapes

    def get_wires(self):
        return self.n_wires
