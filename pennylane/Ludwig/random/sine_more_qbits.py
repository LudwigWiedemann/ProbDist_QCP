import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# Define problem parameters
num_qubits = 2
num_layers = 2

training_inputs = [0, 0.2, 0.4, 0.6, 0.8, 1.0]


# Define training input and target output pairs
training_data = [
    (training_inputs[0], np.sin(training_inputs[0])),
    (training_inputs[1], np.sin(training_inputs[1])),
    (training_inputs[2], np.sin(training_inputs[2])),
    (training_inputs[3], np.sin(training_inputs[3])),
    (training_inputs[4], np.sin(training_inputs[4])),
    (training_inputs[5], np.sin(training_inputs[5])),
]

# Initialize quantum device and optimizer
dev = qml.device("default.qubit", wires=num_qubits)
params = np.random.rand(num_layers)
opt = qml.GradientDescentOptimizer()

# Define the circuit with two RY gates
@qml.qnode(dev)
def circuit(params, x):
    qml.RY(params[0] * x, wires=0)
    qml.RY(params[1] * x, wires=1)
    return qml.probs(wires=[0, 1])

# Define the mean squared error cost function
def cost(params, x, target):
    probabilities = circuit(params, x)
    # Calculate the average probability
    average_prob = np.mean(probabilities)
    print(average_prob)
    return ((average_prob - target) ** 2) / 2

# Train the circuit using gradient descent
print("Training the circuit...")
for iteration in range(500):
    for x, target in training_data:
        params = opt.step(cost, params, x=x, target=target)

    if iteration % 10 == 0:
        print(f"Iteration {iteration}:")
        for x, target in training_data:
            predicted_output = circuit(params, x)
            error = np.abs(predicted_output - target)
            #print(f"Input: {x}, Expected: {target:.4f}, Predicted: {predicted_output:.4f}, Error: {error:.4f}")

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
