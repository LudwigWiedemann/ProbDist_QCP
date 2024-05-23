import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import numpy as numpy

num_qubits = 1
num_layers = 9
num_params_per_layer = 1
total_num_params = num_layers * num_params_per_layer

num_training_points = 50  # Increase the number of training points
training_inputs = np.linspace(0, 10, num_training_points)  # Use np.linspace for even distribution
training_iterations = 200
noise = 0.1*numpy.random.randn(len(training_inputs))

dev = qml.device("default.qubit", wires=num_qubits)
opt = qml.GradientDescentOptimizer(0.001)

def f(x):
    return np.sin(x) + noise

# plot f(x)
plt.plot(training_inputs, f(training_inputs))
plt.show()