import traceback

import pennylane as qml
import numpy as np


# Define the target function (modify as needed)
def target_function(x):
    return np.sin(x)


# Generate sample data
x_data = np.linspace(0, 2 * np.pi, 100)
y_data = target_function(x_data)


# Quantum circuit definition
def qml_circuit(params, wires):
    """
    This circuit applies Ry rotations to each qubit based on parameters.
    """
    for i, param in enumerate(params):
        qml.RY(param, wires[i])

    # Add your desired quantum operations here (e.g., entangling gates)

    # Measurement
    return [qml.expval(qml.PauliZ(wire)) for wire in wires]


# Cost function (Mean Squared Error)
def cost_function(params, dev):
    """
    Calculates mean squared error between predicted and target function values.
    """
    # Execute the quantum circuit with current parameters
    circuit = qml.QNode(qml_circuit, wires=dev.wires, device=dev)
    y_pred = circuit(params)

    # Calculate mean squared error
    return np.mean((y_data - y_pred) ** 2)


# VQC Optimization
def run_vqc(n_qubits, n_iters):
    """
    Performs VQC optimization for function approximation.

    Args:
        n_qubits: Number of qubits in the quantum circuit.
        n_iters: Number of optimization iterations.
    """
    try:
        # Define device and number of qubits
        dev = qml.device("default.qubit", wires=n_qubits)

        # Initial parameter guess
        init_params = np.random.rand(n_qubits)

        # Select optimizer (Adam in this case)
        opt = qml.AdamOptimizer(stepsize=0.01)

        # Perform optimization
        for n in range(n_iters):
            params, cost = opt.step(cost_function, init_params, dev=dev)
            init_params = params

        # Print optimized parameters
        print("Optimized Parameters:", params)

        return params
    except Exception as e:
        print(f"Optimization Error: ")
        traceback.print_exc()
        return None


# Run VQC with default parameters (modify as needed)
n_qubits = 2  # maybe [0,1] for 2 qubits or [0] for 1 qubit
n_iters = 100
optimized_params = run_vqc(n_qubits, n_iters)

if optimized_params is not None:
    # Evaluation (if optimization successful)
    dev = qml.device("default.qubit", wires=n_qubits)
    circuit = qml.QNode(qml_circuit, wires=dev.wires, device=dev)

    new_x = np.linspace(0, np.pi * 2, 50)  # Test data points
    y_pred = target_function(new_x)  # True target values
    predicted_y = circuit(optimized_params)

    # Print comparison between predicted and target values
    print("Predicted vs. Target:")
    for x, y_t, y_p in zip(new_x, y_pred, predicted_y):
        print(f"x: {x:.2f}, Target: {y_t:.4f}, Predicted: {y_p:.4f}")
else:
    print("VQC optimization failed.")
