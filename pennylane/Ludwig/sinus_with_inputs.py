import pennylane as qml
from pennylane import numpy as np

# Define problem parameters
num_qubits = 1
num_layers = 2

# Define training input and target output pairs
training_data = [
    (754.954, np.sin(754.954 * np.pi / 180)),  # Add more input-target pairs here
    # ...
]

# Initialize quantum device and optimizer
dev = qml.device("default.qubit", wires=num_qubits)
params = np.random.rand(num_layers)
opt = qml.GradientDescentOptimizer()


# Define the circuit with two RY gates
@qml.qnode(dev)
def circuit(params, x):
    qml.RY(params[0] * x, wires=0)
    qml.RY(params[1] * x, wires=0)
    return qml.expval(qml.PauliZ(wires=0))


# Define the mean squared error cost function
def cost(params, x, target):
    predicted_output = circuit(params, x)
    return ((predicted_output - target) ** 2) / 2


# Train the circuit using gradient descent
print("Training the circuit...")
for iteration in range(200):
    for x, target in training_data:
        params = opt.step(cost, params, x=x, target=target)  # Use x and target here

    if iteration % 10 == 0:
        average_error = 0
        for x, target in training_data:
            predicted_output = circuit(params, x)
            average_error += ((predicted_output - target) ** 2)
        average_error /= len(training_data)
        print(f"Iteration {iteration}: Average Error = {average_error}")

# Evaluate the trained circuit for new input values
print("Evaluating the trained circuit...")
for x in [30, 50, 100]:  # You can change these values
    optimized_output = circuit(params, x)
    actual_output = np.sin(x * np.pi / 180)
    print(f"Optimized Output for x={x}: {optimized_output}")
    print(f"Actual Output: {actual_output}")
    print(f"Error: {np.abs(optimized_output - actual_output)}")
