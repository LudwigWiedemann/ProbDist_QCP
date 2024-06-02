import circuit as cir
import matplotlib.pyplot as plt
from pennylane import numpy as np


class Evaluation:

    def __init__(self, mean, min, max):
        self.mean_y = mean
        self.y_min = min
        self.y_max = max


def plot(param_list, distributions, f):
    x_axis = np.linspace(0, 20, 200)
    shots = []
    firstparam = True
    for params in param_list:
        shot = [cir.run_circuit(params, x) for x in x_axis]
        shots.append(shot)
        plt.plot(x_axis, shot, alpha=0.1)
        # if firstparam:
        # else:
        #     plt.plot(x_axis, shot[1], 'g--', alpha=0.1)
        # firstparam = False

    evaluations = []
    for i in range(len(x_axis)):
        mean_y = 0
        min_y = 100
        max_y = -100

        for shot in shots:
            mean_y += shot[i]
            if shot[i] < min_y:
                min_y = shot[i]

            if shot[i] > max_y:
                max_y = shot[i]
        mean_y /= len(shots)
        eval = Evaluation(mean_y, min_y, max_y)
        evaluations.append(eval)

    mean = []
    minima = []
    maxima = []
    for eval in evaluations:
        mean.append(eval.mean_y)
        minima.append(eval.y_min)
        maxima.append(eval.y_max)

    #plt.plot(x_axis, mean, label="mean", alpha=0.7)
    plt.fill_between(x_axis, minima, maxima, alpha=0.3)
    # plt.plot(x_axis, minima, label="minima", alpha=0.3)
    # plt.plot(x_axis, maxima, label="maxima", alpha=0.3)

    firstdist = True
    # for dist in distributions:
    #     training_x = [data[0] for data in dist]
    #     training_y = [data[1] for data in dist]
    #     if firstdist:
    #         plt.scatter(training_x, training_y, s=5, label="data points", alpha=0.1)
    #     else:
    #         plt.scatter(training_x, training_y, s=5, alpha=0.1)
    #     firstdist = False

    true_outputs = f(x_axis)
    plt.plot(x_axis, true_outputs, label="Actual f(x)", alpha=0.9)

    plt.ylim(-1, 1)
    plt.grid(True)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title(str(len(distributions)) + " Shots")
    plt.show()
