import pennylane as qml
import matplotlib.pyplot as plt
from main_pipline.input.div.logger import logger
import main_pipline.input.div.filemanager as file
import main_pipline.input.div.config_manager as config
from pathlib import Path
from pennylane import numpy as np
from abc import ABC, abstractmethod

path= Path(file.path).parent.joinpath("\Circuits")

# TODO adapt circuit model so it works with tensorflow architecture
# TODO split circuits into single files, perhaps a manager for generating new ones
class ICircuit(ABC):
    def __init__(self, num_qubits=4, num_layers=1, num_shots=1, num_weights_per_layer=1):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.num_weights_per_layer = num_weights_per_layer
        self.num_weights = num_layers * num_weights_per_layer
        self.training_device = qml.device("default.qubit", wires=num_qubits)
        self.prediction_device = qml.device("default.qubit", wires=num_qubits, shots=num_shots)

    def print_circuit(self, circuit_function, *args):
        print(qml.draw(circuit_function)(*args))
        qml.drawer.use_style("black_white")
        fig, ax = qml.draw_mpl(circuit_function)(*args)
        config.circuit_used="ICircuit"
        plt.savefig(f"{path}\ICircuit-{file.time_started}.png")  #saves Circuit.png
        plt.show()

    @abstractmethod
    def run_without_shots(self):
        def _circuit(weights, inputs):
            pass
        pass

    def run_with_shots(self):
        def _circuit(weights, inputs):
            pass
        pass


class RY_Circuit(ICircuit):
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

class RYXZ_Circuit(ICircuit):
    def __init__(self):
        super().__init__(num_qubits=5, num_layers=5) # add changes inside the brackets

    # prediction_device = qml.device("default.qubit", wires=1, shots=1)
    training_device = qml.device("default.qubit", wires=1)
    num_layers = 2
    num_weights_per_layer = 3
    num_weights = num_layers * num_weights_per_layer
    def run_without_shots(self):
        @qml.qnode(self.training_device)
        def _circuit(weights, inputs):
            for i in range(self.num_layers):
                qml.RY(inputs, wires=0)
                qml.RX(weights[3 * i], wires=0)
                qml.RY(weights[3 * i + 1], wires=0)
                qml.RZ(weights[3 * i + 2], wires=0)
                config.circuit_used="RYXZ_Circuit without shots"
            return qml.expval(qml.PauliZ(wires=0))
        return _circuit
    def run_with_shots(self):
        @qml.qnode(self.prediction_device)
        def _circuit(weights, inputs):
            for i in range(self.num_layers):
                qml.RY(inputs, wires=0)
                qml.RX(weights[3 * i], wires=0)
                qml.RY(weights[3 * i + 1], wires=0)
                qml.RZ(weights[3 * i + 2], wires=0)
                config.circuit_used="RYXZ_Circuit with shots"
            return qml.expval(qml.PauliZ(wires=0))
        return _circuit

    def print_circuit(self, circuit_function, *args):
        pass
    # evaluation

class Entangled_circuit:
    prediction_device = qml.device("default.qubit", wires=1, shots=1)
    training_device = qml.device("default.qubit", wires=1)
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
            config.circuit_used="Entangled_Circuit"
            return qml.expval(qml.PauliZ(wires=0))
        return _circuit

# def create_tape():
#
#
# def gen_QNode():
#     ops = []
#     ops.append(qml.RX(0))
#     ops.append(qml.RY(0))
#     return ops
# def target():
#     for op in ops:
#         op.queue()
#     return qml.state()
#     return target


#if __name__ == '__main__':
    #circuit = Entangled_circuit
    #circuit.create()([], 1)
    #abstract_circuit = RY_Circuit.print_circuit()




