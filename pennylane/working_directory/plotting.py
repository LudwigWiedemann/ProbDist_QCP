import circuit as cir
import matplotlib.pyplot as plt
from pennylane import numpy as np


def plot(param_list, distributions, f):

    x_axis = np.linspace(0, 20, 200)
    shots = []
    firstparam = True
    for params in param_list:
        predicted_outputs = [cir.run_circuit(params, x) for x in x_axis]
        shots.append(predicted_outputs)
        if firstparam:
            plt.plot(x_axis, predicted_outputs, 'g--', label="Predicted Sin", alpha=0.1)
        else:
            plt.plot(x_axis, predicted_outputs, 'g--', alpha=0.1)
        firstparam = False

    mean = [0.0] * len(x_axis)
    for i in range(len(x_axis)):
        for shot in shots:
            mean[i] += shot[i]
        mean[i] = mean[i] / len(shots)
    plt.plot(x_axis, mean, label="mean", alpha=0.7)


    firstdist = True
    for dist in distributions:
        training_x = [data[0] for data in dist]
        training_y = [data[1] for data in dist]
        if firstdist:
            plt.scatter(training_x, training_y, s=5, label="data points", alpha=0.1)
        else:
            plt.scatter(training_x, training_y, s=5, alpha=0.1)
        firstdist = False

    true_outputs = f(x_axis)
    plt.plot(x_axis, true_outputs, label="Actual f(x)", alpha=0.3)

    plt.ylim(-1, 1)
    plt.grid(True)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(str(len(distributions))+" Shots")
    plt.show()
