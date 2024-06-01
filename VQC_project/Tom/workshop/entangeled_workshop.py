import pennylane as qml



dev = qml.device("default.qubit", wires=config['n_qubits'])
@qml.qnode(dev, interface='tf')
def quantum_circuit(inputs, weights):
    qml.AngleEmbedding(inputs, wires=range(config['n_qubits']))
    qml.StronglyEntanglingLayers(weights, wires=range(config['n_qubits']))
    return qml.expval(qml.PauliZ(0))





