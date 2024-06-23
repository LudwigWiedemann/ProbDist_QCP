import pennylane as qml
import main_pipline.input.div.config_manager as config
from main_pipline.models_circuits_and_piplines.circuits.old_circuits.ICircuit import ICircuit

class RYCircuit(ICircuit):

    def run_without_shots(self):
        @qml.qnode(self.training_device)
        def _circuit(weights, inputs):
            qml.RY(weights[0] * inputs, wires=0)
            qml.RY(weights[1] * inputs, wires=0)
            qml.RY(weights[2], wires=0)
            qml.RY(weights[3], wires=0)
            qml.RY(weights[4] * inputs, wires=0)
            qml.RY(weights[5], wires=0)
            qml.RY(weights[6], wires=0)
            qml.RY(weights[7], wires=0)
            qml.RY(weights[8], wires=0)
            config.circuit_used="RY_Circuit without shots"
            return qml.expval(qml.PauliZ(wires=0))
        return _circuit

    def run_with_shots(self):
        @qml.qnode(self.prediction_device)
        def _circuit(weights, inputs):
            qml.RY(weights[0] * inputs, wires=0)
            qml.RY(weights[1] * inputs, wires=0)
            qml.RY(weights[2], wires=0)
            qml.RY(weights[3], wires=0)
            qml.RY(weights[4] * inputs, wires=0)
            qml.RY(weights[5], wires=0)
            qml.RY(weights[6], wires=0)
            qml.RY(weights[7], wires=0)
            qml.RY(weights[8], wires=0)
            config.circuit_used="RY_Circuit with shots"
            return qml.expval(qml.PauliZ(wires=0))
        return _circuit
