import circuit as cir
import matplotlib.pyplot as plt
from pennylane import numpy as np


def plot(param_list, distributions, f):

    x_axis = np.linspace(0, 20, 200)

    for params in param_list:
        predicted_outputs = [cir.run_circuit(params, x) for x in x_axis]
        plt.plot(x_axis, predicted_outputs, label="Predicted Sin", alpha=0.1)

    for dist in distributions:
        training_x = [data[0] for data in dist]
        training_y = [data[1] for data in dist]
        plt.scatter(training_x, training_y, s=5, label="data points", alpha=0.1)

    true_outputs = f(x_axis)
    plt.plot(x_axis, true_outputs, label="Actual f(x)", alpha=0.3)

    plt.ylim(-1, 1)
    plt.grid(True)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Shots")
    plt.show()
