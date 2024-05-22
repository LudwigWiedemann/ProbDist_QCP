import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# Define problem parameters
num_qubits = 1
num_layers = 5

num_training_points = 30  # Increase the number of training points
training_inputs = np.linspace(0, 3*np.pi, num_training_points)  # Use np.linspace for even distribution
training_data = [(x, np.sin(x)) for x in training_inputs]


# Initialize quantum device and optimizer
dev = qml.device("default.qubit", wires=num_qubits)
params = np.random.rand(num_layers)
learning_rate = 0.02  # Reduced learning rate
opt = qml.AdamOptimizer(learning_rate)
# Define the circuit with two RY gates
@qml.qnode(dev)
def circuit(params, x):
    qml.RY(params[0] * x, wires=0)
    qml.RY(params[1] * x, wires=0)
    qml.RY(params[2] * x, wires=0)
    qml.RY(params[3] * x, wires=0)
    qml.RY(params[4] * x, wires=0)

    return qml.expval(qml.PauliZ(wires=0))

# Define the mean squared error cost function
def cost(params, x, target):
    predicted_output = circuit(params, x)
    return ((predicted_output - target) ** 2) / 2

# Train the circuit using gradient descent
print("Training the circuit...")
for iteration in range(200):
    for x, target in training_data:
        params = opt.step(cost, params, x=x, target=target)

    if iteration % 10 == 0:
        print(f"Iteration {iteration}:")
        for x, target in training_data:
            predicted_output = circuit(params, x)
            error = np.abs(predicted_output - target)
            print(f"Input: {x}, Expected: {target:.4f}, Predicted: {predicted_output:.4f}, Error: {error:.4f}")

# Evaluate the trained circuit and generate plot
print("Evaluating the trained circuit...")
x_values = np.linspace(-5*np.pi, 10*np.pi, 100)  # Define range for plotting
actual_sin = np.sin(x_values)
predicted_outputs = [circuit(params, x) for x in x_values]
plt.ylim(-2, 2)
plt.grid(True)
plt.plot(x_values, actual_sin, label="Actual Sin")
plt.plot(x_values, predicted_outputs, label="Predicted Sin")
plt.legend()
plt.xlabel("x")
plt.ylabel("Sin(x)")
plt.title("Actual vs. Predicted Sin")
plt.show()
