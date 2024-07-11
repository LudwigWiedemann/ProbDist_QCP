import pennylane as qml
num_shots = 100
device_with_shots = qml.device("default.qubit", wires=1, shots=num_shots)
device = qml.device("default.qubit", wires=1)

@qml.qnode(device)
def run_without_shots():
    qml.Hadamard(wires=0)
    eval = qml.expval(qml.PauliZ(wires=0))
    return eval
@qml.qnode(device_with_shots)
def run_with_shots():
    qml.Hadamard(wires=0)
    eval = qml.expval(qml.PauliZ(wires=0))
    return eval

reps = 10
print("========================================================================================================================")
print("No Shots - " + str(reps) + " predictions (always the same result for same params):")
for i in range(reps):
    print(": " + str(run_without_shots(params, x)[1]))