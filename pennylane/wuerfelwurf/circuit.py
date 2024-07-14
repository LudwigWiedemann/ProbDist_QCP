import pennylane as qml

num_wires = 3
num_shots = 20
num_layers = 2
shot_dev = qml.device("default.qubit", wires=num_wires, shots=num_shots)


# shot_dev = qml.device("default.qubit", wires=num_wires)

@qml.qnode(shot_dev)
def predict_wuerfelwurf(weights):
    num_qubits = num_wires
    params_per_qubit = 3  # Assuming RZ, RX, RY for each qubit
    total_params_needed = num_qubits * params_per_qubit * 2  # For two layers

    if len(weights) != total_params_needed:
        raise ValueError(f"Expected {total_params_needed} weights, got {len(weights)}")

    # First parameterized layer
    for qubit in range(num_qubits):
        qml.RZ(weights[qubit * params_per_qubit + 0], wires=qubit)
        qml.RX(weights[qubit * params_per_qubit + 1], wires=qubit)
        qml.RY(weights[qubit * params_per_qubit + 2], wires=qubit)

    # Entangling layer
    for qubit in range(num_qubits - 1):
        qml.CNOT(wires=[qubit, qubit + 1])
    qml.CNOT(wires=[num_qubits - 1, 0])  # Entangle the last qubit with the first for a loop

    # Second parameterized layer
    offset = num_qubits * params_per_qubit  # Offset for the second layer weights
    for qubit in range(num_qubits):
        qml.RZ(weights[offset + qubit * params_per_qubit + 0], wires=qubit)
        qml.RX(weights[offset + qubit * params_per_qubit + 1], wires=qubit)
        qml.RY(weights[offset + qubit * params_per_qubit + 2], wires=qubit)

    return qml.expval(qml.PauliZ(wires=0))


@qml.qnode(shot_dev)
def poc(weights):
    qml.Hadamard(wires=0)
    qml.RY(weights[0], wires=0)
    qml.RX(weights[1], wires=0)
    qml.RY(weights[2], wires=0)
    qml.RX(weights[3], wires=0)
    # qml.RY(weights[3], wires=0)
    # qml.RY(weights[4], wires=0)
    # qml.RY(weights[5], wires=0)
    return qml.expval(qml.PauliZ(wires=0))
