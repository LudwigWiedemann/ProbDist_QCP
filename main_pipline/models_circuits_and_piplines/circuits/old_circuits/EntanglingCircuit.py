import pennylane as qml
import main_pipline.input.div.config_manager as config
from main_pipline.models_circuits_and_piplines.circuits.old_circuits.ICircuit import ICircuit


class EntanglingCircuit(ICircuit):

    def create(self):
        """
        Returns a quantum node which represents a quantum circuit with predefined entangling layers
        :param weights: weights used to optimize inputs
        :param inputs: from training data
        :return: Qnode of the circuit
        """

        @qml.qnode(self.dev)
        def _circuit(weights, inputs):
            qml.AngleEmbedding(inputs, wires=range(self.num_qubits))
            qml.StronglyEntanglingLayers(weights, wires=range(self.num_qubits))
            config.circuit_used = "Entangled_Circuit"
            return qml.expval(qml.PauliZ(wires=0))

        return _circuit
