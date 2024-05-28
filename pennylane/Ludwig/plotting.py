import circuit as cir
import matplotlib.pyplot as plt
from pennylane import numpy as np


def plot(params, training_data, f):
    x_axis = np.linspace(-10 * np.pi, 10 * np.pi, 100)

    predicted_outputs = [cir.run_circuit(params, x) for x in x_axis]
    true_outputs = f(x_axis)

    plt.ylim(-2, 2)
    plt.grid(True)
    training_x = [data[0] for data in training_data]
    training_y = [data[1] for data in training_data]

    plt.scatter(training_x, training_y)
    plt.plot(x_axis, true_outputs, label="Actual f(x)")
    plt.plot(x_axis, predicted_outputs, label="Predicted Sin")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("Sin(x)")
    plt.title("Actual vs. Predicted Sin")
    plt.show()
