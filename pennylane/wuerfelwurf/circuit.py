import pennylane as qml

num_wires = 3
num_shots = 20
num_layers = 2
shot_dev = qml.device("default.qubit", wires=num_wires, shots=num_shots)


# shot_dev = qml.device("default.qubit", wires=num_wires)

@qml.qnode(shot_dev)
def predict_wuerfelwurf(weights):
    for layer in range(num_layers):
        qml.RZ(weights[layer + 0], wires=0)
        qml.RX(weights[layer + 1], wires=0)
        qml.RY(weights[layer + 2], wires=0)

        qml.RZ(weights[layer + 3], wires=1)
        qml.RX(weights[layer + 4], wires=1)
        qml.RY(weights[layer + 5], wires=1)

        qml.RZ(weights[layer + 6], wires=2)
        qml.RX(weights[layer + 7], wires=2)
        qml.RY(weights[layer + 8], wires=2)

        for wire in range(num_wires):
            for other_wire in range(num_wires):
                if wire != other_wire:
                    qml.CNOT(wires=[wire, other_wire])
    return qml.expval(qml.PauliZ(wires=0))


@qml.qnode(shot_dev)
def poc(weights):
    qml.RY(weights[0], wires=0)
    qml.RX(weights[1], wires=0)
    qml.RY(weights[2], wires=0)
    qml.RX(weights[3], wires=0)
    # qml.RY(weights[3], wires=0)
    # qml.RY(weights[4], wires=0)
    # qml.RY(weights[5], wires=0)
    return qml.expval(qml.PauliZ(wires=0))
