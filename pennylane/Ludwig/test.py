import pennylane as qml
from pennylane import numpy as np

# Wähle ein Quantum Device
dev = qml.device('default.qubit', wires=1)

# Definiere eine Quantenfunktion (QNode)
@qml.qnode(dev)
def circuit():
    qml.Hadamard(wires=0)  # Beispiel: Ein Hadamard-Gatter auf Qubit 0
    return qml.expval(qml.PauliZ(0))  # Erwartungswert der Pauli-Z-Operation

# Führe den QNode aus, um den Erwartungswert zu erhalten
expectation_value = circuit()

print("Erwartungswert:", expectation_value)
