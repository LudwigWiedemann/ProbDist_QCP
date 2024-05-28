import pennylane as qml

dev = qml.device("default.qubit", wires=[0, 1])


def my_first_circuit(theta):
    qml.Hadamard(wires=0)
    qml.CNOT(wires=[0, 1])
    qml.RY(theta, wires=0)

    return qml.probs(wires=[0, 1])


my_first_QNode = qml.QNode(my_first_circuit, dev)

print(my_first_QNode(0.54))

dev = qml.device("default.qubit", wires=3)
