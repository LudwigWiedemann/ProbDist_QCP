import pennylane as qml#
from pennylane import numpy as np


class Circuits:

    def __init__(self, num_qubits, num_layers):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit", wires=num_qubits)

    def ry_circuit(self, weights, inputs):
        @qml.qnode(self.dev)
        def circuit(weights, inputs):
            qml.RY(weights[0] * inputs, wires=0)
            qml.RY(weights[1] * inputs, wires=0)
            qml.RY(weights[2], wires=0)
            qml.RY(weights[3], wires=0)
            qml.RY(weights[4] * inputs, wires=0)
            qml.RY(weights[5], wires=0)
            qml.RY(weights[6], wires=0)
            qml.RY(weights[7], wires=0)
            qml.RY(weights[8], wires=0)
            return qml.expval(qml.PauliZ(wires=0))

        return circuit(weights, inputs)

    def entangling_circuit(self, weights, inputs):
        @qml.qnode(self.dev)
        def circuit(weights, inputs):
            qml.AngleEmbedding(inputs, wires=range(self.num_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))
            return qml.expval(qml.PauliZ(0))

        return circuit(weights, inputs)

    def print_circuits(self, circuit_function):
        circuit = qml.QNode(circuit_function, self.dev)
        print(qml.draw(circuit)(np.pi/4, 0.7))

if __name__ == '__main__':
    circuits = Circuits(1, 1)

    circuits.print_circuits(circuits.ry_circuit)
    circuits.print_circuits(circuits.entangling_circuit)
    print("Circuits printed successfully!")