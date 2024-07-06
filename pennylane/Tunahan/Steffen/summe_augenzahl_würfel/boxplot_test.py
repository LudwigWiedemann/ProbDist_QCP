# -*- coding: utf-8 -*-

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# Anzahl der Qubits und Shots (Messungen)
num_qubits = 4
shots = 10000

# Definieren des devices
dev = qml.device('default.qubit', wires=num_qubits, shots=shots)

# Variational Quantum Circuit (VQC)
def circuit(weights):
    for i in range(num_qubits):
        qml.Rot(*weights[i], wires=i)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])

@qml.qnode(dev)
def quantum_circuit(weights):
    circuit(weights)
    return qml.probs(wires=range(num_qubits))

# Zielwahrscheinlichkeiten
target_probs = np.array([1/36, 2/36, 3/36, 4/36, 5/36, 6/36, 5/36, 4/36, 3/36, 2/36, 1/36])

# Indexe der relevanten Zustaende fuer die Zielwahrscheinlichkeiten
relevant_indices = np.arange(11) + 2  # Augensummen 2 bis 12

# Verlustfunktion
def cost(weights):
    probs = quantum_circuit(weights)
    relevant_probs = probs[relevant_indices]
    return np.sum((relevant_probs - target_probs) ** 2)

# Optimierung
opt = qml.AdamOptimizer(stepsize=0.1)
weights = np.random.rand(num_qubits, 3)

# Training
num_steps = 100
for step in range(num_steps):
    weights = opt.step(cost, weights)
    if (step + 1) % 10 == 0:
        current_cost = cost(weights)
        print(f'Step {step+1}: Cost = {current_cost:.4f}')

# Ergebnis
probs = quantum_circuit(weights)
relevant_probs = probs[relevant_indices]

print("Ziel-Wahrscheinlichkeiten: ", target_probs)
print("Erlernte Wahrscheinlichkeiten: ", relevant_probs)
print("Summe der quanten Wahrscheinlichkeiten: ", np.sum(probs))
print("Summe der erlernten Wahrscheinlichkeiten: ", np.sum(relevant_probs))

# Boxplot erstellen
x_labels = np.arange(2, 13)
data_for_boxplot = [probs[i] for i in relevant_indices]

plt.figure(figsize=(8, 6))
plt.boxplot(data_for_boxplot, labels=x_labels[1:])
plt.xlabel('Summe der Augenzahlen')
plt.ylabel('Wahrscheinlichkeit')
plt.title(f'Boxplot der Wahrscheinlichkeitsverteilung der Augensummen mit {shots} Shots')
plt.grid(True)
plt.show()
