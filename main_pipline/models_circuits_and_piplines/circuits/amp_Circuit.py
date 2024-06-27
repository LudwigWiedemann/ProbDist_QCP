import math
from collections import defaultdict
from functools import partial
from itertools import chain

import pennylane as qml
import numpy as np


class AmpCircuit:
    def __init__(self, shots=None):
        self.shots = shots

    def run(self, inputs, weights_0, weights_1, weights_2):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_weights(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_wires(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def draw_circuit(self, filename="circuit_diagram.txt"):
        inputs = np.random.random(2 ** self.n_wires)
        weights = [np.random.random(self.weight_shapes[key]) for key in self.weight_shapes]
        circuit = self.run()
        drawer = qml.draw(circuit)
        diagram = drawer(inputs, *weights)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(diagram)
        print(f"Circuit diagram saved to {filename}")


class base_Amp_Circuit(AmpCircuit):
    def __init__(self, config, shots=None):
        super().__init__()
        self.n_wires = config['future_steps']
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
        self.n_wires = config['future_steps']
        self.layer = 3
        self.ops = ['RY', 'RX', 'RZ']
        self.weight_shapes = defaultdict(int)
        for i in range(self.layer):
            for name in self.ops:
                self.weight_shapes[f"{name}_{i}"] = self.n_wires

    def run(self):
        training_device = qml.device("default.qubit", wires=self.n_wires)

        @partial(qml.batch_input, argnum=0)
        @qml.qnode(training_device, interface='tf')
        def circuit(inputs, RY_0, RX_0, RZ_0, RY_1, RX_1, RZ_1, RX_2, RY_2, RZ_2):
            qml.AmplitudeEmbedding(features=inputs, wires=range(self.n_wires), normalize=True, pad_with=0.)
            qml.broadcast(qml.CNOT, wires=range(self.n_wires), pattern="chain")
            qml.broadcast(qml.RY, wires=range(self.n_wires), pattern="single", parameters=RY_0)
            qml.broadcast(qml.RX, wires=range(self.n_wires), pattern="single", parameters=RX_0)
            qml.broadcast(qml.RZ, wires=range(self.n_wires), pattern="single", parameters=RZ_0)
            qml.broadcast(qml.CNOT, wires=range(self.n_wires), pattern="all_to_all")

            qml.broadcast(qml.RZ, wires=range(self.n_wires), pattern="single", parameters=RZ_1)
            qml.broadcast(qml.RX, wires=range(self.n_wires), pattern="single", parameters=RX_1)
            qml.broadcast(qml.RY, wires=range(self.n_wires), pattern="single", parameters=RY_1)
            qml.broadcast(qml.CNOT, wires=range(self.n_wires), pattern="all_to_all")

            qml.broadcast(qml.RY, wires=range(self.n_wires), pattern="single", parameters=RY_2)
            qml.broadcast(qml.RZ, wires=range(self.n_wires), pattern="single", parameters=RZ_2)
            qml.broadcast(qml.RX, wires=range(self.n_wires), pattern="single", parameters=RX_2)
            qml.broadcast(qml.CNOT, wires=range(self.n_wires), pattern="all_to_all")

            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

        return circuit

    def get_weights(self):
        return self.weight_shapes

    def get_wires(self):
        return self.n_wires


class tangle_Amp_Circuit(AmpCircuit):
    def __init__(self, config):
        super().__init__()
        self.shots = config['shots']
        self.n_wires = config['future_steps']
        self.weight_shapes = {"weights": (config['layers'], self.n_wires, 3)}

    def run(self):
        training_device = qml.device("default.qubit", wires=self.n_wires, shots=self.shots)

        @partial(qml.batch_input, argnum=0)
        @qml.qnode(training_device, interface='tf')
        def circuit(inputs, weights):
            qml.AmplitudeEmbedding(features=inputs, wires=range(self.n_wires), normalize=True, pad_with=0.)
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_wires))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

        return circuit

    def get_weights(self):
        return self.weight_shapes

    def get_wires(self):
        return self.n_wires


class double_Amp_Circuit(AmpCircuit):
    def __init__(self, config):
        super().__init__()
        self.n_wires = config['future_steps'] * 2
        self.weight_shapes = {
            "strong_layer_0": (3, self.n_wires // 2, 3),
            "Rot_0": (self.n_wires * 3),
            "Rot_1": (self.n_wires * 3),
            "Rot_2": (self.n_wires * 3),
            # "double_layer_1": (3 * self.n_wires * 3),
            # "strong_layer_1": (3, self.n_wires // 2, 3)
        }

    def double_layer(self, Rot_0, Rot_1, Rot_2):
        qml.broadcast(qml.CNOT, pattern='chain', wires=list(range(self.n_wires))[0::2])
        qml.broadcast(qml.Rot, pattern='single', wires=range(self.n_wires), parameters=Rot_0)
        qml.broadcast(qml.CNOT, pattern='double', wires=range(self.n_wires))
        qml.broadcast(qml.Rot, pattern='single', wires=range(self.n_wires), parameters=Rot_1)
        qml.broadcast(qml.CY, pattern='double', wires=range(self.n_wires))
        qml.broadcast(qml.Rot, pattern='single', wires=range(self.n_wires), parameters=Rot_2)
        qml.broadcast(qml.CZ, pattern='double', wires=range(self.n_wires))

    def run(self):
        training_device = qml.device("default.qubit", wires=self.n_wires)

        @partial(qml.batch_input, argnum=0)
        @qml.qnode(training_device, interface='tf')
        def circuit(inputs, strong_layer_0, Rot_0, Rot_1, Rot_2):
            qml.AmplitudeEmbedding(features=inputs, wires=list(range(self.n_wires))[0::2], normalize=True, pad_with=0.0)
            self.double_layer(Rot_0, Rot_1, Rot_2)
            qml.StronglyEntanglingLayers(strong_layer_0, wires=list(range(self.n_wires))[0::2])
            # qml.broadcast(qml.Rot, pattern='single', wires=range(self.n_wires), parameters=weights[0])
            # self.double_layer(double_layer_1)
            # qml.StronglyEntanglingLayers(strong_layer_1, wires=list(range(self.n_wires))[0::2])
            return [qml.expval(qml.PauliZ(i)) for i in list(range(self.n_wires))[0::2]]

        return circuit

    def get_weights(self):
        return self.weight_shapes

    def get_wires(self):
        return self.n_wires


class test_Amp_Circuit(AmpCircuit):
    def __init__(self, config):
        super().__init__()
        self.n_wires = config['future_steps']
        self.weight_shapes = {"weights": self.n_wires}

    def run(self):
        training_device = qml.device("default.qubit", wires=self.n_wires)

        @partial(qml.batch_input, argnum=0)
        @qml.qnode(training_device, interface='tf')
        def circuit(inputs, weights):
            qml.AmplitudeEmbedding(features=inputs, wires=range(self.n_wires), normalize=True, pad_with=0.)
            qml.broadcast(qml.RX, range(self.n_wires), pattern='single', parameters=weights)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

        return circuit

    def get_weights(self):
        return self.weight_shapes

    def get_wires(self):
        return self.n_wires
