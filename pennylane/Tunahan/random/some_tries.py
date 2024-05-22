import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# Define the quantum device
n_qubits = 2
dev = qml.device('default.qubit', wires=n_qubits)


# Define the variational quantum circuit
def circuit(params, x):
    qml.templates.AngleEmbedding(x, wires=range(n_qubits))
    qml.templates.StronglyEntanglingLayers(params, wires=range(n_qubits))
    return qml.expval(qml.PauliZ(0))


# Initialize random parameters
n_layers = 3
params = np.random.randn(n_layers, n_qubits, 3)

# Define the quantum node
qnode = qml.QNode(circuit, dev)

# Generate training data
X = np.linspace(0, 2 * np.pi, 50)
Y = np.sin(X)


# Define the cost function
def cost(params):
    predictions = [qnode(params, x) for x in X]
    return np.mean((predictions - Y) ** 2)


# Optimize the circuit parameters
opt = qml.GradientDescentOptimizer(stepsize=0.1)
n_iterations = 200

for i in range(n_iterations):
    params = opt.step(cost, params)
    if i % 20 == 0:
        print(f'Iteration {i}: cost = {cost(params):.4f}')

# Evaluate the trained model
predictions = [qnode(params, x) for x in X]

# Plot the results
plt.figure(figsize=(8, 6))
plt.plot(X, Y, label='True sine function')
plt.plot(X, predictions, label='Quantum circuit approximation')
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.legend()
plt.show()
