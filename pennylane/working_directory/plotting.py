import circuit as cir
import matplotlib.pyplot as plt
import numpy as np


def plot(param_list, distributions, f, seed):

    x_axis = np.linspace(0, 20, 200)

    firstparam = True
    for params in param_list:
        predicted_outputs = [cir.run_seeded_circuit(params, x, seed) for x in x_axis]
        if firstparam:
            plt.plot(x_axis, predicted_outputs, 'g--', label="Predicted Sin", alpha=0.1)
        else:
            plt.plot(x_axis, predicted_outputs, 'g--', alpha=0.1)
        firstparam = False


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
