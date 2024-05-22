import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# Set a random seed for reproducibility
np.random.seed(42)

# Define the quantum device
n_qubits = 1
dev = qml.device('default.qubit', wires=n_qubits)

# Define the parameterized quantum circuit (ansatz)
@qml.qnode(dev)
def quantum_circuit(params, x):
    qml.RX(x, wires=0)
    qml.RY(params[0], wires=0)
    qml.RZ(params[1], wires=0)
    return qml.expval(qml.PauliZ(0))

# Define the cost function
def cost(params, x, y):
    predictions = np.array([quantum_circuit(params, xi) for xi in x])
    return np.mean((predictions - y) ** 2)

# Generate training data from the sine function
x_train = np.linspace(0, 2 * np.pi, 50)
y_train = np.sin(x_train)

# Initialize the parameters
params = np.random.rand(2)

# Set up the optimizer
opt = qml.AdamOptimizer(stepsize=0.1)
n_steps = 100

# Training loop
for step in range(n_steps):
    params, cost_val = opt.step_and_cost(lambda p: cost(p, x_train, y_train), params)
    if step % 10 == 0:
        print(f'Step {step}: Cost = {cost_val:.4f}')

# Plot the results
x_test = np.linspace(0, 2 * np.pi, 100)
y_test = np.sin(x_test)
predictions = np.array([quantum_circuit(params, xi) for xi in x_test])

plt.plot(x_train, y_train, 'bo', label='Training data (sin(x))')
plt.plot(x_test, predictions, 'r-', label='VQC predictions')
plt.legend()
plt.xlabel('x')
plt.ylabel('sin(x)')
plt.title('Variational Quantum Circuit Approximation of sin(x)')
plt.show()
