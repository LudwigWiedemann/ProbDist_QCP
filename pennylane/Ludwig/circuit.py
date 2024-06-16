import pennylane as qml
from pennylane import numpy as np


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


dev = qml.device("default.qubit", wires=3)


@qml.qnode(dev)
def multiple_wires(params, inputs):
    for i in range(len(inputs)):
        qml.RY(params[i] * inputs[i], wires=i)
    for e in range(len(inputs) - 1):
        qml.CNOT(wires=[e, 2])
    return qml.expval(qml.PauliZ(wires=2))


# p = np.random.rand(10)
# i = [0,1,2,3,4,5,6,7,8,9]
# print(multiple_wires(p, i))
