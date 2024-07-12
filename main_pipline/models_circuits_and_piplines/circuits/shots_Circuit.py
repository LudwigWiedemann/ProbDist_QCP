import math
from collections import defaultdict
from functools import partial

import pennylane as qml
from pennylane import numpy as np


class Shot_Circuit:
    def __init__(self, shots=None):
        self.shots = shots

    def run(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_weights(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_wires(self):
        raise NotImplementedError("This method should be implemented by subclasses.")


class test_Shot_Circuit(Shot_Circuit):
    def __init__(self, config):
        super().__init__()
        self.shots = config["shots"]
        self.n_wires = config['future_steps']
        self.weight_shapes = {"weights": self.n_wires}

    def run(self):
        @partial(qml.batch_input, argnum=0)
        @qml.qnode(qml.device("default.qubit", wires=self.n_wires), interface='tf')
        def circuit(inputs, weights):
            qml.AmplitudeEmbedding(features=inputs, wires=range(self.n_wires), normalize=True, pad_with=0.)
            qml.broadcast(qml.RX, range(self.n_wires), pattern='single', parameters=weights)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

        return circuit

    def run_shot(self, custome_shots=None):
        if custome_shots is None:
            shots = self.shots
        else:
            shots = custome_shots

        @qml.qnode(qml.device("default.qubit", wires=self.n_wires, shots=shots), interface=None)
        def circuit(inputs, weights):
            qml.AmplitudeEmbedding(features=inputs, wires=range(self.n_wires), normalize=True, pad_with=0.)
            qml.broadcast(qml.RX, range(self.n_wires), pattern='single', parameters=weights)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

        return circuit

    def get_weights(self):
        return self.weight_shapes

    def get_wires(self):
        return self.n_wires


class Tangle_Shot_Circuit(Shot_Circuit):
    def __init__(self, config):
        super().__init__()
        self.shots = config["shots"]
        self.n_wires = config['future_steps']
        self.weight_shapes = {"weights": (config['layers'], self.n_wires, 3)}

    def run(self):

        @partial(qml.batch_input, argnum=0)
        @qml.qnode(qml.device("default.qubit", wires=self.n_wires), interface='tf')
        def circuit(inputs, weights):
            qml.AmplitudeEmbedding(features=inputs, wires=range(self.n_wires), normalize=True, pad_with=0.)
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_wires))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

        return circuit

    def run_shot(self, custom_shots=None):
        if custom_shots is None:
            shots = self.shots
        else:
            shots = custom_shots

        @qml.qnode(qml.device("default.qubit", wires=self.n_wires, shots=shots), interface=None,
                   expansion_strategy="device")
        def circuit(inputs, weights):
            qml.AmplitudeEmbedding(features=inputs, wires=range(self.n_wires), normalize=True, pad_with=0.)
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_wires))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

        return circuit

    def get_weights(self):
        return self.weight_shapes

    def get_wires(self):
        return self.n_wires

    def draw_circuit(self, weights, filename="circuit_diagram.txt"):
        inputs = np.random.random(2 ** self.n_wires)
        circuit = self.run_shot()
        drawer = qml.draw(circuit)
        diagram = drawer(inputs, weights)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(diagram)
        print(f"Circuit diagram saved to {filename}")

        fig, ax = qml.draw_mpl(circuit)(inputs, weights)
        fig.show()


class Ludwig_Shot_Circuit(Shot_Circuit):
    def __init__(self, config):
        super().__init__()
        self.shots = config["shots"]
        self.n_wires = config['future_steps']
        self.ops = ['RY', 'RX', 'RZ']
        self.weight_shapes = defaultdict(int)
        for i in range(3):
            for name in self.ops:
                self.weight_shapes[f"{name}_{i}"] = self.n_wires

    def run(self):

        @partial(qml.batch_input, argnum=0)
        @qml.qnode(qml.device("default.qubit", wires=self.n_wires), interface='tf')
        def circuit(inputs, RY_0, RX_0, RZ_0, RY_1, RX_1, RZ_1, RY_2, RX_2, RZ_2):
            qml.AmplitudeEmbedding(features=inputs, wires=range(self.n_wires), normalize=True)
            qml.RZ(RZ_0, wires=0)
            qml.RX(RX_0, wires=0)
            qml.RY(RY_0, wires=0)

            qml.RZ(RZ_1, wires=1)
            qml.RX(RX_1, wires=1)
            qml.RY(RY_1, wires=1)

            qml.RZ(RZ_2, wires=2)
            qml.RX(RX_2, wires=2)
            qml.RY(RY_2, wires=2)

            for wire in range(self.n_wires):
                for other_wire in range(self.n_wires):
                    if wire != other_wire:
                        qml.CNOT(wires=[wire, other_wire])

            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

        return circuit

    def run_shot(self, custom_shots=None):
        if custom_shots is None:
            shots = self.shots
        else:
            shots = custom_shots

        @qml.qnode(qml.device("default.qubit", wires=self.n_wires, shots=shots), interface=None,
                   expansion_strategy="device")
        def circuit(inputs, weights):
            qml.AmplitudeEmbedding(features=inputs, wires=range(self.n_wires), normalize=True, pad_with=0.)
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_wires))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

        return circuit

    def get_weights(self):
        return self.weight_shapes

    def get_wires(self):
        return self.n_wires

    def draw_circuit(self, weights, filename="circuit_diagram.txt"):
        inputs = np.random.random(2 ** self.n_wires)
        circuit = self.run_shot()
        drawer = qml.draw(circuit)
        diagram = drawer(inputs, weights)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(diagram)
        print(f"Circuit diagram saved to {filename}")

        fig, ax = qml.draw_mpl(circuit)(inputs, weights)
        fig.show()


class Ludwig2_Shot_Circuit(Shot_Circuit):
    def __init__(self, config):
        super().__init__()
        self.shots = config["shots"]
        self.n_wires = int(math.log2(config['time_steps']))
        self.layers = config['layers']
        self.outputs = config['future_steps']
        self.weight_shapes = {"weights": (self.layers, self.n_wires, 3)}

    def run(self):

        @partial(qml.batch_input, argnum=0)
        @qml.qnode(qml.device("default.qubit", wires=self.n_wires), interface='tf')
        def circuit(inputs, weights):
            qml.AmplitudeEmbedding(features=inputs, wires=range(self.n_wires), normalize=True)

            for layer in range(self.layers):
                for i in range(self.n_wires):
                    qml.RY(weights[layer][i][0], wires=i)
                    qml.RZ(weights[layer][i][1], wires=i)
                # entangle the output wires with all other ones
                output_wires = range(self.outputs)
                for wire in output_wires:
                    for i in range(self.n_wires):
                        if not i == wire:
                            qml.CRY(weights[layer][i][2], wires=[i, wire])

                outputs = []
                for wire in output_wires:
                    outputs.append(qml.expval(qml.PauliZ(wires=wire)))
                return outputs

        return circuit

    def run_shot(self, custom_shots=None):
        if custom_shots is None:
            shots = self.shots
        else:
            shots = custom_shots

        @qml.qnode(qml.device("default.qubit", wires=self.n_wires, shots=shots), interface=None)
        def circuit(inputs, weights):
            qml.AmplitudeEmbedding(features=inputs, wires=range(self.n_wires), normalize=True)
            for layer in range(self.layers):
                for i in range(self.n_wires):
                    qml.RY(weights[0][layer][i][0], wires=i)
                    qml.RZ(weights[0][layer][i][1], wires=i)
                # entangle the output wires with all other ones
                output_wires = range(self.outputs)
                for wire in output_wires:
                    for i in range(self.n_wires):
                        if not i == wire:
                            qml.CRY(weights[0][layer][i][2], wires=[i, wire])

                outputs = []
                for wire in output_wires:
                    outputs.append(qml.expval(qml.PauliZ(wires=wire)))
                return outputs

        return circuit

    def get_weights(self):
        return self.weight_shapes

    def get_wires(self):
        return self.n_wires

    def draw_circuit(self, weights, filename="circuit_diagram.txt"):
        inputs = np.random.random(2 ** self.n_wires)
        circuit = self.run_shot()
        drawer = qml.draw(circuit)
        diagram = drawer(inputs, weights)
        with open(filename, "w", encoding="utf-8") as f:
            f.write(diagram)
        print(f"Circuit diagram saved to {filename}")

        fig, ax = qml.draw_mpl(circuit)(inputs, weights)
        fig.show()


class Reup_Shot_Circuit(Shot_Circuit):
    def __init__(self, config):
        super().__init__()
        self.shots = config["shots"]
        self.n_wires = config['future_steps']
        self.weight_shapes = {"weights": (config['layers'] // 3, self.n_wires, 3),
                              "weights_2": (config['layers'] // 3, self.n_wires, 3),
                              "weights_3": (config['layers'] // 3, self.n_wires, 3)}

    def run(self):
        @partial(qml.batch_input, argnum=0)
        @qml.qnode(qml.device("default.qubit", wires=self.n_wires), interface='tf')
        def circuit(inputs, weights, weights_2, weights_3):
            angel_index = []
            for wires in range(self.n_wires):
                angel_index.append(2 ** self.n_wires // self.n_wires * wires)
            qml.AmplitudeEmbedding(features=inputs, wires=range(self.n_wires), normalize=True, pad_with=0.)
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_wires))
            qml.AngleEmbedding(features=inputs[angel_index], wires=range(self.n_wires))
            qml.StronglyEntanglingLayers(weights_2, wires=range(self.n_wires))
            qml.AngleEmbedding(features=inputs[angel_index], wires=range(self.n_wires))
            qml.StronglyEntanglingLayers(weights_3, wires=range(self.n_wires))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

        return circuit

    def run_shot(self, custom_shots=None):
        if custom_shots is None:
            shots = self.shots
        else:
            shots = custom_shots

        @qml.qnode(qml.device("default.qubit", wires=self.n_wires, shots=shots), interface=None)
        def circuit(inputs, weights, weights_2, weights_3):
            angel_index = []
            for wires in range(self.n_wires):
                angel_index.append(2 ** self.n_wires // self.n_wires * wires)
            qml.AmplitudeEmbedding(features=inputs, wires=range(self.n_wires), normalize=True, pad_with=0.)
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_wires))
            qml.AngleEmbedding(features=inputs[angel_index], wires=range(self.n_wires))
            qml.StronglyEntanglingLayers(weights_2, wires=range(self.n_wires))
            qml.AngleEmbedding(features=inputs[angel_index], wires=range(self.n_wires))
            qml.StronglyEntanglingLayers(weights_3, wires=range(self.n_wires))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

        return circuit

    def get_weights(self):
        return self.weight_shapes

    def get_wires(self):
        return self.n_wires

    def draw_circuit(self, weights, filename="circuit_diagram.txt"):
        inputs = np.random.random(2 ** self.n_wires)
        circuit = self.run_shot()
        drawer = qml.draw(circuit)
        diagram = drawer(inputs, weights[0], weights[1], weights[2])
        with open(filename, "w", encoding="utf-8") as f:
            f.write(diagram)
        print(f"Circuit diagram saved to {filename}")

        fig, ax = qml.draw_mpl(circuit)(inputs, weights[0], weights[1], weights[2])
        fig.show()


class Custom_Shot_Circuit(Shot_Circuit):
    def __init__(self, config):
        super().__init__()
        self.shots = config["shots"]
        self.n_wires = config['future_steps']
        self.weight_shapes = {"weights": 1, "weights2": 1}

    def run(self):
        @partial(qml.batch_input, argnum=0)
        @qml.qnode(qml.device("default.qubit", wires=self.n_wires), interface='tf')
        def circuit(inputs, weights):
            qml.AmplitudeEmbedding(features=inputs, wires=range(self.n_wires), normalize=True, pad_with=0.)
            qml.RX(weights, 0)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

        return circuit

    def run_shot(self):
        @qml.qnode(qml.device("default.qubit", wires=self.n_wires, shots=self.shots), interface=None)
        def circuit(inputs, weights):
            qml.AmplitudeEmbedding(features=inputs, wires=range(self.n_wires), normalize=True, pad_with=0.)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

        return circuit

    def get_weights(self):
        return self.weight_shapes

    def get_wires(self):
        return self.n_wires
