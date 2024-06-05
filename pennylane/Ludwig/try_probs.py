from pennylane import pennylane as qml

num_shots = 3
device = qml.device("default.qubit", wires=1, shots=10)
@qml.qnode(device)
def run_circuit():
    qml.Hadamard(wires=0)
    return qml.sample(qml.PauliZ(wires=0))


print("Result: " + str(run_circuit()))








