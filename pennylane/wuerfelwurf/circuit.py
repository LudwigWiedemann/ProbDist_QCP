import pennylane as qml
import numpy as np

num_wires = 3
num_shots = 200
num_layers = 1
shot_dev = qml.device("default.qubit", wires=num_wires, shots=num_shots)


# shot_dev = qml.device("default.qubit", wires=num_wires)

@qml.qnode(shot_dev)
def predict_wuerfelwurf(weights):
    # Assuming weights is a flat array with length num_wires * 3 * num_layers
    # and num_layers = 1 for simplicity
    for i in range(num_wires):
        qml.RY(weights[i * 3], wires=i)
        qml.RX(weights[i * 3 + 1], wires=i)
        qml.RY(weights[i * 3 + 2], wires=i)

    # Apply PauliZ to each wire individually
    return [qml.sample(qml.PauliZ(wires=i)) for i in range(num_wires)]

def interpret_measurement(measurement):
    # Convert the list to a NumPy array before flattening
    measurement_array = np.array(measurement)
    binary_result = ''.join(['1' if x == 1 else '0' for x in measurement_array.flatten()])
    number = int(binary_result, 2)
    return number



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
