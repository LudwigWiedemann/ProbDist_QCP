import pennylane as qml
from pennylane import numpy as np

# Define problem parameters
num_qubits = 1
num_layers = 5
example_input = 1
target_output = np.sin(example_input)
print("trying to approximate sin(" + str(example_input) + ")")

# Initialize quantum device and optimizer
dev = qml.device("default.qubit", wires=num_qubits)
params = np.random.rand(num_layers)
print(params)
opt = qml.GradientDescentOptimizer()

# Define the circuit with two RY gates
@qml.qnode(dev)
def circuit(params):
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=0)
    qml.RY(params[2], wires=0)
    qml.RY(params[3], wires=0)
    qml.RY(params[4], wires=0)
    return qml.expval(qml.PauliZ(0))

# Define the mean squared error cost function
def cost(params):
    predicted_output = circuit(params)
    return ((predicted_output - target_output) ** 2) / 2

# Train the circuit using gradient descent
print("Training the circuit...")
for iteration in range(200):
    params = opt.step(cost, params)
    if iteration % 10 == 0:
        predicted_output = circuit(params)
        print(f"Iteration {iteration}: Predicted Output = {predicted_output}")

# Evaluate the trained circuit
print("Evaluating the trained circuit...")
optimized_output = circuit(params)
print(f"Optimized Output: {optimized_output}")
print("compared to actual result sin(x): " + str(np.sin(example_input)))
print("Abweichung: " + str(np.abs(optimized_output - target_output)))
