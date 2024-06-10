import pennylane as qml
import main_pipline.input.div.config_manager as config

class BaseCircuit:
    def __init__(self):
        super().__init__()

    def run(self, inputs, weights_0, weights_1, weights_2):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_weights(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def get_wires(self):
        raise NotImplementedError("This method should be implemented by subclasses.")

class new_RYXZ_Circuit(BaseCircuit):
    def __init__(self):
        super().__init__()
        self.weight_shapes = {"weights_0": 1, "weights_1": 1, "weights_2": 1}
        self.n_wires = 1

    def run(self):
        training_device = qml.device("default.qubit", wires=self.n_wires)

        @qml.qnode(training_device, interface='tf')
        def _circuit(inputs, weights_0, weights_1, weights_2):
            for i in range(5):
                qml.RY(inputs[0], wires=0)
                qml.RX(weights_0, wires=0)
                qml.RY(weights_1, wires=0)
                qml.RZ(weights_2, wires=0)
                config.circuit_used= "variable-RYXZ-Circuit"
            return qml.expval(qml.PauliZ(wires=0))

        return _circuit

    def get_weights(self):
        return self.weight_shapes

    def get_wires(self):
        return self.n_wires
