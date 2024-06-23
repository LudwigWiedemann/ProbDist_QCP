import pennylane as qml
import matplotlib.pyplot as plt
import main_pipline.input.div.filemanager as file
import main_pipline.input.div.config_manager as config
from abc import ABC, abstractmethod
from pathlib import Path

# TODO get the values for qubits, layers, shots, weights_per_layer from the configs
class ICircuit(ABC):
    def __init__(self, num_qubits=4, num_layers=1, num_shots=1, num_weights_per_layer=1):
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.num_shots = num_shots
        self.num_weights_per_layer = num_weights_per_layer
        self.num_weights = num_layers * num_weights_per_layer
        self.training_device = qml.device("default.qubit", wires=num_qubits)
        self.prediction_device = qml.device("default.qubit", wires=num_qubits, shots=num_shots)

    @abstractmethod
    def circuit(self, weights, inputs):
        NotImplementedError("This method should be implemented by subclasses.")

    def run(self, inputs, weights, use_shots=True):
        if use_shots:
            circuit_with_shots = qml.QNode(self.circuit, self.prediction_device)
            return circuit_with_shots(weights, inputs)
        else:
            circuit_without_shots = qml.QNode(self.circuit, self.training_device)
            return circuit_without_shots(weights, inputs)

    def print_circuit(self, circuit_function, *args):
        print(qml.draw(circuit_function)(*args))
        qml.drawer.use_style("black_white")
        fig, ax = qml.draw_mpl(circuit_function)(*args)

        # Config Stuff from Pat
        path = Path(file.path).parent.joinpath("\Circuits")
        config.circuit_used = "ICircuit"
        plt.savefig(f"{path}\ICircuit-{file.time_started}.png")  #saves Circuit.png

        plt.show()

    def print_self_variables(self):
        print(f"num_qubits: {self.num_qubits}")
        print(f"num_layers: {self.num_layers}")
        print(f"num_shots: {self.num_shots}")
        print(f"num_weights_per_layer: {self.num_weights_per_layer}")
        print(f"num_weights: {self.num_weights}")
        print(f"training_device: {self.training_device}")
        print(f"prediction_device: {self.prediction_device}")

