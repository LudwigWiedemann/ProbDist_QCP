import pennylane as qml
from pennylane import numpy as np

# VQC with variable rotations around the Y axis
def circuit(params, x):
    qml.RY(params[0], wires=0)  # Variational parameter
    qml.RY(params[1], wires=0)  # Variational parameter
    return qml.expval(qml.PauliZ(wires=0))  # Measure Pauli-Z

# Cost function (mean squared error between predicted and actual values)
def cost(params, x, y):
    predicted_y = circuit(params, x)
    return np.mean((predicted_y - y)**2)

# Training data (x, sin(x)) pairs
training_data = [(x, np.sin(x)) for x in np.linspace(0, 10, 100)]  # 100 data points

# Random initial guess for parameters
params = np.random.rand(2)

# Optimizer (gradient descent)
optimizer = qml.AdamOptimizer(stepsize=0.01)

# Training loop (optimize parameters to minimize cost)
for _ in range(1000):
    for x, y in training_data:
        params, _ = optimizer.step(cost, params, x=x, y=y)

# Evaluate the trained circuit
x_new = 5  # New input value
predicted_y = circuit(params, x_new)
print(f"Predicted value for sin({x_new}):", predicted_y)
