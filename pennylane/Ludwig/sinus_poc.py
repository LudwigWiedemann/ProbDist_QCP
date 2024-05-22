import pennylane as qml
from pennylane import numpy as np

# Define number of Qbits and layers for the device
num_qubits = 1
num_layers = 2  # Increased circuit depth

# Define example input and target output
example_input = np.pi
target_output = np.sin(example_input)

# Initialize PennyLane-QuantumDevice
dev = qml.device("default.qubit", wires=num_qubits)

# Initialize parameters for the circuit
params = np.random.rand(num_layers)
opt = qml.GradientDescentOptimizer()

# Define the circuit with two layers of RY gates
@qml.qnode(dev)
def circuit(params, x=None):
    qml.RY(params[0], wires=0)
    qml.RY(params[1], wires=0)
    return qml.expval(qml.PauliZ(0))

# Define the cost function using mean squared error
def cost(params):
    predicted_output = circuit(params, x=example_input)
    return ((predicted_output - target_output) ** 2) / 2  # Mean squared error

print("Target output:", target_output)

# Train the circuit using gradient descent
for i in range(100):
    params = opt.step(cost, params)
    if i % 10 == 0:
        print("Step", i, "Cost:", cost(params))

# Evaluate the trained circuit
optimized_output = circuit(params, x=example_input)
print("Optimized output:", optimized_output)
