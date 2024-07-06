import pennylane as qml
from pennylane import numpy as np

def get_circuit():
    @qml.qnode(qml.device("default.qubit", wires=6, shots=5), interface = None)
    def circuit2(inputs, weights):
        qml.AmplitudeEmbedding(features=inputs, wires=range(6), normalize=True, pad_with=0.)
        # qml.broadcast(qml.RX, range(6), pattern='single', parameters=weights)
        qml.RX(weights, 0)
        return [qml.expval(qml.PauliZ(i)) for i in range(6)]
    return circuit2
circuit = get_circuit()

