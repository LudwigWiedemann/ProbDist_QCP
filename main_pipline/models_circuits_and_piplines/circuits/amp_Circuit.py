import math
from collections import defaultdict
from functools import partial
import pennylane as qml
import numpy as np


class AmpCircuit:
    def __init__(self):
        super().__init__()

    def run(self, inputs, weights_0, weights_1, weights_2):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_weights(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_wires(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def draw_circuit(self, filename="circuit_diagram.txt"):
        """Draw the quantum circuit and save it to a file."""
        # Generate dummy inputs and weights for drawing the circuit
        inputs = np.random.random(2 ** self.n_wires)  # Correct input length for AmplitudeEmbedding
        weights = [np.random.random(self.weight_shapes[key]) for key in self.weight_shapes]
        circuit = self.run()
        drawer = qml.draw(circuit)
        diagram = drawer(inputs, *weights)
        # Save the diagram to a file
        with open(filename, "w", encoding="utf-8") as f:
            f.write(diagram)
        print(f"Circuit diagram saved to {filename}")


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
            qml.AmplitudeEmbedding(features=inputs, wires=range(self.n_wires), normalize=True)
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

class Tangle_Amp_Circuit(AmpCircuit):
    def __init__(self, config):
        super().__init__()
        self.n_wires = int(math.log2(config['time_steps']))
        self.weight_shapes = {"weights": (24, self.n_wires, 3)}

    def run(self):
        training_device = qml.device("default.qubit", wires=self.n_wires)

        @partial(qml.batch_input, argnum=0)
        @qml.qnode(training_device, interface='tf')
        def circuit(inputs, weights):
            qml.AmplitudeEmbedding(features=inputs, wires=range(self.n_wires), normalize=True)
            qml.StronglyEntanglingLayers(weights, wires=range(self.n_wires))
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]

        return circuit

    def get_weights(self):
        return self.weight_shapes

    def get_wires(self):
        return self.n_wires

class Test_Circuit(AmpCircuit):
    def __init__(self, config):
        super().__init__()
        self.n_wires = int(math.log2(config['time_steps']))
        self.weight_shapes = {"weights": 1}

    def run(self):
        training_device = qml.device("default.qubit", wires=self.n_wires)

        @partial(qml.batch_input, argnum=0)
        @qml.qnode(training_device, interface='tf')
        def circuit(inputs, weights):
            qml.AmplitudeEmbedding(features=inputs, wires=range(self.n_wires), normalize=True)
            qml.broadcast(qml.RX,range(self.n_wires), pattern='single', parameters=weights)
            return [qml.expval(qml.PauliZ(i)) for i in range(self.n_wires)]
        return circuit

    def get_weights(self):
        return self.weight_shapes

    def get_wires(self):
        return self.n_wires



if __name__ == "__main__":
    config = {
        'time_steps': 64
    }

    base_circuit = base_Amp_Circuit(config)
    print("Base Circuit:")
    base_circuit.draw_circuit("base_circuit_diagram.txt")

    layered_circuit = layered_Amp_Circuit(config)
    print("Layered Circuit:")
    layered_circuit.draw_circuit("layered_circuit_diagram.txt")