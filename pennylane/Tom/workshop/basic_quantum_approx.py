import time

import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt


# Define the target function
def target_function(x):
    return np.sin(x) + 0.5 * np.cos(2 * x) + 0.25 * np.sin(3 * x)


# Generate training data
def generate_training_data(num_points=50, range_start=0, range_end=10):
    x_train = np.linspace(range_start, range_end, num_points)
    y_train = target_function(x_train)
    return x_train, y_train


# Create a more complex quantum circuit
def create_quantum_circuit(num_qubits=1, num_layers=3):
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def quantum_circuit(params, x):
        qml.AngleEmbedding(features=[x], wires=range(num_qubits))
        qml.StronglyEntanglingLayers(params, wires=range(num_qubits))
        return qml.expval(qml.PauliZ(0))

    return quantum_circuit


# Define the cost function
def cost(circuit, params, x_train, y_train):
    predictions = np.array([circuit(params, x) for x in x_train])
    return np.mean((y_train - predictions) ** 2)


# Train the quantum circuit
def train_quantum_circuit(circuit, params, x_train, y_train, num_epochs=300, stepsize=0.1):
    opt = qml.AdamOptimizer(stepsize=stepsize)
    for epoch in range(num_epochs):
        params, cost_val = opt.step_and_cost(lambda v: cost(circuit, v, x_train, y_train), params)
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: cost = {cost_val.item()}")
            print(f"Params: {params}")
    return params


# Evaluate the trained circuit
def evaluate_circuit(circuit, params, x_train, y_train, x_range, y_true_func):
    x_test = np.linspace(x_range[0], x_range[1], 100)
    y_test = y_true_func(x_test)
    y_pred = np.array([circuit(params, x) for x in x_test])

    plt.figure(figsize=(10, 6))
    plt.plot(x_test, y_test, label='True Function', color='blue')
    plt.plot(x_test, y_pred, label='VQC Approximation', color='red')
    plt.scatter(x_train, y_train, color='green', label='Training Points')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Quantum Circuit Learning Complex Function using PennyLane')
    plt.legend()
    plt.grid(True)
    plt.show()


# Main function to execute the training and evaluation
def main():
    start_time = time.time()
    num_qubits = 1
    num_layers = 3  # Increase the number of layers for more complexity
    num_training_points = 50
    training_iterations = 300

    # Generate training data
    x_train, y_train = generate_training_data(num_points=num_training_points)

    # Initialize parameters
    params_shape = (num_layers, num_qubits, 3)
    params = np.random.uniform(-np.pi, np.pi, params_shape)
    params = np.array(params, requires_grad=True)  # Ensure gradient tracking

    # Create quantum circuit
    circuit = create_quantum_circuit(num_qubits=num_qubits, num_layers=num_layers)

    # Train the circuit
    trained_params = train_quantum_circuit(circuit, params, x_train, y_train, num_epochs=training_iterations)

    # Evaluate the circuit
    evaluate_circuit(circuit, trained_params, x_train, y_train, x_range=(-3 * np.pi, 6 * np.pi),
                     y_true_func=target_function)
    print(f"Model trained in {time.time() - start_time} seconds")


# Execute the main function
if __name__ == "__main__":
    main()
