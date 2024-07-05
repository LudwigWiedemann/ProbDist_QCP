from functools import partial

import numpy as np
import pennylane as qml


class Shot_Circuit:
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
            #qml.RX(weights, wires=0)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

        return circuit

    def shot_circuit(self):
        @qml.qnode(qml.device("default.qubit", wires=self.n_wires, shots=self.shots), interface=None)
        def circuit(inputs, weights):
            qml.AmplitudeEmbedding(features=inputs, wires=range(self.n_wires), normalize=True, pad_with=0.)
            qml.broadcast(qml.RX, range(self.n_wires), pattern='single', parameters=weights)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

        return circuit

    def get_weights(self):
        return self.weight_shapes

    def get_wires(self):
        return self.n_wires
