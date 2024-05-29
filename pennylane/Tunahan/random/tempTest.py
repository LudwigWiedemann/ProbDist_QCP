import pennylane as qml
import circuits as cs
import matplotlib as plt

from pennylane.Tunahan.main import QuantumMlAlgorithm


def circuit(self, params, x):
    circuits = cs.Circuits(self.num_qubits, self.num_layers)

    @qml.qnode(self.dev)
    def _circuit(params, x):
        return circuits.ry_circuit(params, x)

    return _circuit(params, x)

# plot the circuit outcome for sin(x)
