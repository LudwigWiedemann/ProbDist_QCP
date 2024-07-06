import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

np.random.seed(404)
num_qubits = 4
shots = 10000
dev = qml.device('default.qubit', wires=num_qubits, shots=shots)

# Vqc
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

# Indexe der relevanten zustaende fuer die zielwahrscheinlichkeiten
# Eig haben wir 16 ergebnisse im vqc aber nur 12 sind relevant bei zwei Wuerfeln
relevant_indices = np.arange(11) + 2  # Augensummen 2 bis 12

# Optimierung
def cost(weights):
    probs = quantum_circuit(weights)
    relevant_probs = probs[relevant_indices]
    return np.sum((relevant_probs - target_probs) ** 2)

opt = qml.AdamOptimizer(stepsize=0.1)  # Krasser unterschied zu GradientDescentOptimizer
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
# Interessant zu wissen da normalerweise ja 1
print("Summe der erlernten Wahrscheinlichkeiten: ", np.sum(relevant_probs))

# Plotting
plt.figure(figsize=(8, 6))
plt.boxplot(relevant_probs)
plt.xticks(np.arange(1, len(relevant_indices) + 1), np.arange(2, 13))
plt.xlabel('Summe der Augenzahlen')
plt.ylabel('Wahrscheinlichkeit')
plt.title(f'Boxplot der Wahrscheinlichkeitsverteilung der Augensummen mit {shots} Shots')
plt.show()
