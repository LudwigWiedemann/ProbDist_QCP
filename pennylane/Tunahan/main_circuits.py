import matplotlib.pyplot as plt
import pennylane as qml
from pennylane import numpy as np

# OLD VERSION STARTS HERE
num_qubits = 1
num_layers = 9
device = qml.device("default.qubit", wires=num_qubits)


@qml.qnode(device)
def run_circuit(params, x):
    qml.RY(params[0] * x, wires=0)
    qml.RY(params[1] * x, wires=0)
    qml.RY(params[2], wires=0)
    qml.RY(params[3], wires=0)
    qml.RY(params[4] * x, wires=0)
    qml.RY(params[5], wires=0)
    qml.RY(params[6], wires=0)
    qml.RY(params[7], wires=0)
    qml.RY(params[8], wires=0)
    return qml.expval(qml.PauliZ(wires=0))


# OLD VERSION ENDS HERE

# NEW VERSION:
class Circuits:
    """
    A class used to represent a Quantum Circuit

    ...

    Attributes
    ----------
    num_qubits : int
        number of qubits in the circuit
    num_layers : int
        #EXPERIEMTAL WE DON'T REALLY USE IT number of layers in the circuit
    dev : Device
        a PennyLane device to run the circuit

    Methods
    -------
    ry_circuit(weights, inputs):
        Returns a quantum node which represents a quantum circuit with only RY rotation gates
    entangling_circuit(weights, inputs):
        Returns a quantum node which represents a quantum circuit with predefined entangling layers
    print_circuits(circuit_function) #EXPERIMENTAL DOESN'T WORK FOR NOW:
        Prints the circuit diagram of a given circuit function
    """

    def __init__(self, num_qubits=4, num_layers=1):
        """
        Constructs all the necessary attributes for the Circuits object when initialized.

        Parameters
        ----------
            num_qubits : int
                number of qubits in the circuit
            num_layers : int
                number of layers in the circuit
        """
        self.num_qubits = num_qubits
        self.num_layers = num_layers
        self.dev = qml.device("default.qubit")

    def ry_circuit(self):
        """
        Returns a quantum node which represents a quantum circuit with only RY rotation gates
        :param weights: weights used to optimize inputs
        :param inputs: from training data
        :return: Qnode of the circuit
        """

        @qml.qnode(self.dev)
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
            return qml.expval(qml.PauliZ(wires=0))

        return _circuit

    def ry_circuit_prob(self):
        """
        Returns a quantum node which represents a quantum circuit with only RY rotation gates
        :param weights: weights used to optimize inputs
        :param inputs: from training data
        :return: Qnode of the circuit
        """

        @qml.qnode(self.dev, shots=100)
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
            return qml.expval(qml.PauliZ(wires=0))

        return _circuit

    def entangling_circuit(self):
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
            return qml.expval(qml.PauliZ(wires=0))

        return _circuit

    def random_circuit(self):
        @qml.qnode(self.dev)
        def _circuit(x, z):
            qml.QFT(wires=(0, 1, 2, 3))
            qml.IsingXX(1.234, wires=(0, 2))
            qml.Toffoli(wires=(0, 1, 2))
            mcm = qml.measure(1)
            mcm_out = qml.measure(2)
            qml.CSWAP(wires=(0, 2, 3))
            qml.RX(x, wires=0)
            qml.cond(mcm, qml.RY)(np.pi / 4, wires=3)
            qml.CRZ(z, wires=(3, 0))
            return qml.expval(qml.Z(0)), qml.probs(op=mcm_out)

        return _circuit

    def ludwig_circuit(self):
        @qml.qnode(self.dev)
        def _circuit(weights, inputs):
            for i in range(self.num_layers):
                qml.RY(inputs, wires=0)
                qml.RX(weights[3 * i], wires=0)
                qml.RY(weights[3 * i + 1], wires=0)
                qml.RZ(weights[3 * i + 2], wires=0)

            return qml.expval(qml.PauliZ(wires=0))
        return _circuit

    def print_circuit(self, circuit_function, *args):
        """
        Prints the circuit diagram of a given circuit function

        Parameters
        ----------
        :param circuit_function : Any
            a function that returns a quantum node
        :param args:
            arguments to pass to the circuit function
        """
        # Execute the circuit function with the provided arguments
        circuit_result = circuit_function(*args)
        print(f"Result for given arguments is: {circuit_result}")

        # Draw the circuit using qml.draw
        print(qml.draw(circuit_function)(*args))

        # Display the circuit using qml.draw_mpl
        qml.drawer.use_style("black_white")
        fig, ax = qml.draw_mpl(circuit_function)(*args)
        plt.show()
        print("---------------------")


if __name__ == '__main__':
    """
    Test the Circuits class and try to print the circuits.
    """
    circuits = Circuits(num_layers=3)

    # create random inputs number using pennylane numpy.random.random

    weights_ry, weights_entangling, input_1d, inputs_entangling = (
        np.random.random(9),
        np.random.random((1, 4, 3)),
        np.random.random(),
        np.random.random(4)
    )
    weights_ludwig = np.random.random(20)

    circuits.print_circuit(circuits.ry_circuit(), weights_ry, input_1d)
    #circuits.print_circuit(circuits.random_circuit(), input_1d, input_1d)
    circuits.print_circuit(circuits.entangling_circuit(), weights_entangling, inputs_entangling)
    circuits.print_circuit(circuits.ludwig_circuit(), weights_ludwig, input_1d)
