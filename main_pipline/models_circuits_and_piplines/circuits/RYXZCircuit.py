import pennylane as qml
import main_pipline.input.div.config_manager as config
from main_pipline.models_circuits_and_piplines.circuits.ICircuit import ICircuit


class RYXZCircuit(ICircuit):
    def __init__(self, num_qubits=4, num_layers=5, num_shots=10, num_weights_per_layer=1):
        super().__init__(num_qubits, num_layers, num_shots, num_weights_per_layer)

    def circuit(self, weights, inputs):
        for i in range(self.num_layers):
            qml.RY(inputs, wires=0)
            qml.RX(weights[3 * i], wires=0)
            qml.RY(weights[3 * i + 1], wires=0)
            qml.RZ(weights[3 * i + 2], wires=0)
            config.circuit_used="RYXZ_Circuit without shots"
        return qml.expval(qml.PauliZ(wires=0))

    # def run_without_shots(self):
    #     @qml.qnode(self.training_device)
    #     def _circuit(weights, inputs):
    #         for i in range(self.num_layers):
    #             qml.RY(inputs, wires=0)
    #             qml.RX(weights[3 * i], wires=0)
    #             qml.RY(weights[3 * i + 1], wires=0)
    #             qml.RZ(weights[3 * i + 2], wires=0)
    #             config.circuit_used="RYXZ_Circuit without shots"
    #         return qml.expval(qml.PauliZ(wires=0))
    #     return _circuit
    #
    # def run_with_shots(self):
    #     @qml.qnode(self.prediction_device)
    #     def _circuit(weights, inputs):
    #         for i in range(self.num_layers):
    #             qml.RY(inputs, wires=0)
    #             qml.RX(weights[3 * i], wires=0)
    #             qml.RY(weights[3 * i + 1], wires=0)
    #             qml.RZ(weights[3 * i + 2], wires=0)
    #             config.circuit_used="RYXZ_Circuit with shots"
    #         return qml.expval(qml.PauliZ(wires=0))
    #     return _circuit


if __name__ == '__main__':
    circuit = RYXZCircuit(num_layers=1, num_shots=10)
    circuit.print_self_variables()
    circuit.print_circuit([1, 2, 3], 0.5)
