import circuit as cir
import matplotlib.pyplot as plt
from pennylane import numpy as np


def plot(data):
    x_axis = np.linspace(0, len(data), len(data))
    plt.plot(x_axis, data, label="Prediction", alpha=0.5)
    plt.ylim(-2, 2)
    plt.grid(True)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Shots")
    plt.show()


