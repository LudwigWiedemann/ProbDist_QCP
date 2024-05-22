import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
num_qubits = 1
num_layers = 6  # Increased number of layers
learning_rate = 0.001  # Lower learning rate

# Training data
num_training_points = 100  # Increased data points for better accuracy
training_inputs = np.linspace(0, np.pi, num_training_points)  # Evenly spaced inputs
training_data = [(x, np.sin(x)) for x in training_inputs]

# Device and circuit definition
dev = qml.device("default.qubit", wires=num_qubits)


@qml.qnode(dev)
def circuit(params, x):
    for i in range(num_layers):
        qml.RY(params[i * 2] * x, wires=0)  # Apply RX with parameter scaling by x
        qml.RY(params[i * 2 + 1], wires=0)  # Add RY gates for more expressiveness
    return qml.expval(qml.PauliZ(wires=0))


# Cost function (mean squared error)
def cost(params, x, target):
    predicted_output = circuit(params, x)
    return ((predicted_output - target) ** 2) / 2


# Training loop with early stopping
def train_vqc(epochs=200, early_stopping_patience=10):
    best_params = None
    best_error = float("inf")
    early_stopping_counter = 0
    num_params = 2 * num_layers  # Assuming 2 parameters per layer (RX, RY)
    param_range = 2 * np.pi  # Range for random initialization

    params = np.random.uniform(low=-param_range, high=param_range, size=num_params)

    for epoch in range(epochs):
        for x, target in training_data:
            params = opt.step(cost, params, x=x, target=target)

        # Evaluate error after each epoch
        epoch_error = calculate_error(params, training_data)

        if epoch_error < best_error:
            best_params = params.copy()
            best_error = epoch_error
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1

        if early_stopping_counter >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch} due to no improvement in error")
            break

    return best_params


# Error calculation function
def calculate_error(params, training_data):
    total_error = 0.0
    for x, target in training_data:
        predicted_output = circuit(params, x)
        error = np.abs(predicted_output - target)
        total_error += error

    return total_error / len(training_data)


# Training and evaluation
opt = qml.AdamOptimizer(learning_rate)  # Use Adam optimizer
best_params = train_vqc()

# Evaluate the trained circuit on a finer grid
x_values = np.linspace(-np.pi, np.pi, 500)  # More points for smoother plot
predicted_outputs = [circuit(best_params, x) for x in x_values]

plt.ylim(-2, 2)
plt.grid(True)
plt.plot(x_values, np.sin(x_values), label="Actual Sin")
plt.plot(x_values, predicted_outputs, label="Predicted Sin")
plt.legend()
plt.xlabel("x")
plt.ylabel("Sin(x)")
plt.title("Actual vs. Predicted Sin")
plt.show()
