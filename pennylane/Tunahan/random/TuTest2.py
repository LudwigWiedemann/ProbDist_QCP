# import libraries
import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# Define the number of qubits and layers (starting slow)
num_qubits = 1
num_gates = 5
num_layers = 3

# Define the quantum device
dev = qml.device("default.qubit", wires=num_qubits)

# Define the variational circuit
@qml.qnode(dev)
def circuit(weights, x):
    for _ in range(num_layers):
        for i in range(num_qubits):
            qml.RY(weights[i] * x, wires=i)
        qml.broadcast(qml.CNOT, wires=range(num_qubits), pattern="all_to_all")
    return qml.expval(qml.PauliZ(0))

# Generate training data for a function (e.g. sin(x), x^2, sin(x) + x^2, etc.)
num_training_points = 250
input_train = np.random.uniform(0, 4*np.pi, num_training_points) # random values

# Define the target function
sin_x = np.sin(input_train)
sin_x_noise = sin_x + 0.1 * np.random.normal(size=num_training_points)
target_function = sin_x

# Initialize the weights and optimizer
weights = np.random.uniform(0, 2 * np.pi, num_gates)
opt = qml.AdamOptimizer(stepsize=0.01)
steps = 500

# Train the circuit

# Define some loss functions
def mean_squared_error(prediction, target):
    return np.mean((prediction - target)**2)

# Define the cost function
def cost(weights, idx):
    x = input_train[idx]
    target = target_function[idx]
    prediction = circuit(weights, x)
    regularization = np.sum(np.abs(weights))
    return mean_squared_error(prediction, target) + 0.01 * regularization

for i in range(steps):
    for idx in range(num_training_points):
        weights = opt.step(cost, weights, idx=idx)
    if (i + 1) % 5 == 0:
        #print(f"Cost after step {i + 1}/{steps}: {cost(weights, idx)}, \nWeights: \n{weights}")
        print(f"Cost after step {i + 1}/{steps}: {cost(weights, idx=idx)}")

# min_cost = float('inf')
# best_weights = None
# min_cost_step = 0
#
# for idx in range(steps):
#     # Optimize weights using optimizer
#     weights = opt.step(cost, weights, idx=idx)
#
#     # Calculate the cost for the current weights
#     current_cost = cost(weights, idx=idx)
#
#     # If the current cost is less than the minimum cost, update the minimum cost and best weights
#     if current_cost < min_cost:
#         min_cost = current_cost
#         best_weights = weights
#         min_cost_step = idx + 1
#
#     print(f"Cost after step {idx + 1}/{steps}: {current_cost}")
#
# print(f"Best weights that give the minimum cost {min_cost} found in step {min_cost_step} are: \n{best_weights}")

# Plot the predictions
# Generate input values for the plot
input_values = np.array(input_train)

# Compute actual function values and predicted function values
predicted_values = np.array([circuit(weights, x) for x in input_values])

# Sort the values for plotting
sort_indices = np.argsort(input_values)
input_values = input_values[sort_indices]
predicted_values = predicted_values[sort_indices]
target_function = target_function[sort_indices]

# Plot actual function values and predicted function values
plt.figure(figsize=(10, 6))
plt.plot(input_values, target_function, label='Actual function')
plt.plot(input_values, predicted_values, label='Predicted function')
plt.legend()
plt.xlabel('Input')
plt.ylabel('Function values')
plt.title('Actual vs Predicted function')
plt.show()