# -*- coding: utf-8 -*-

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

np.random.seed(404)
num_qubits = 4
shots = 10000

dev = qml.device('default.qubit', wires=num_qubits, shots=shots)

# VQC (ursprünglicher Circuit)
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

# Quantum Circuit für Vergleich
@qml.qnode(dev)
def quantum_circuit_compare(weights):
    # Hier definieren Sie Ihren neuen Circuit, ähnlich wie quantum_circuit
    circuit(weights)
    return qml.probs(wires=range(num_qubits))

# Zielwahrscheinlichkeiten
target_probs = np.array([1/36, 2/36, 3/36, 4/36, 5/36, 6/36, 5/36, 4/36, 3/36, 2/36, 1/36])

# Indexe der relevanten Zustaende für die Zielwahrscheinlichkeiten
relevant_indices = np.arange(11) + 2  # Augensummen 2 bis 12

# Optimierung
def cost(weights):
    probs = quantum_circuit(weights)
    relevant_probs = probs[relevant_indices]
    return np.sum((relevant_probs - target_probs) ** 2)

opt = qml.AdamOptimizer(stepsize=0.1)
weights = np.random.rand(num_qubits, 3)

# Training
num_steps = 100
for step in range(num_steps):
    weights = opt.step(cost, weights)
    if (step + 1) % 10 == 0:
        current_cost = cost(weights)
        print(f'Step {step+1}: Cost = {current_cost:.4f}')

# Ergebnis des ursprünglichen Circuits
probs_learned = quantum_circuit(weights)
relevant_probs_learned = probs_learned[relevant_indices]

# Plotting der gelernten Wahrscheinlichkeitsverteilung
x = np.arange(2, 13)
plt.bar(x - 0.2, target_probs, width=0.4, label='Ziel')
plt.bar(x + 0.2, relevant_probs_learned, width=0.4, label='Erlernt')
plt.xlabel('Summe der Augenzahlen')
plt.ylabel('Wahrscheinlichkeit')
plt.title(f'Verteilung Augensumme zweier Wuerfel mit {shots} Shots (gelernt)')
plt.legend()
plt.show()

# Ausführen und Vergleichen des neuen Circuits
probs_new = quantum_circuit_compare(weights)
relevant_probs_new = probs_new[relevant_indices]

# # Plotting der neuen Wahrscheinlichkeitsverteilung
# plt.bar(x - 0.2, target_probs, width=0.4, label='Ziel')
# plt.bar(x + 0.2, relevant_probs_new, width=0.4, label='Neu')
# plt.xlabel('Summe der Augenzahlen')
# plt.ylabel('Wahrscheinlichkeit')
# plt.title(f'Verteilung Augensumme zweier Wuerfel mit {shots} Shots (neu)')
# plt.legend()
# plt.show()

# Vergleich der Wahrscheinlichkeitsverteilungen
print("Erlernte Wahrscheinlichkeiten: ", relevant_probs_learned)

kl_divergence = np.sum(target_probs * np.log(target_probs / relevant_probs_learned))
print(f"Kullback-Leibler-Divergenz (gelernt vs target): {kl_divergence}")

# kl_divergence_new = np.sum(target_probs * np.log(target_probs / relevant_probs_new))
# print(f"Kullback-Leibler-Divergenz (neu): {kl_divergence_new}")
#
# print(f"Difference: {abs(kl_divergence - kl_divergence_new)}")
