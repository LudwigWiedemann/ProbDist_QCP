import matplotlib.pyplot as plt
from pennylane import numpy as np


def plot(data, x_start, x_end, **kwargs):
    x_axis = np.linspace(x_start, x_end, len(data))
    label= kwargs.get('label',None)
    plt.plot(x_axis, data, label=label, alpha=0.5)
    plt.ylim(-2, 2)
    plt.grid(True)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Shots")
    plt.show()


def f(x):
    return np.sin(x)
    #return np.sin(x) + 0.5 * np.cos(2 * x) + 0.25 * np.sin(3 * x)


x_axis = np.linspace(0, 20, 210)
plt.plot(x_axis, f(x_axis), label="P", alpha=0.5)
plt.ylim(-2, 2)
plt.grid(True)
plt.legend()
plt.xlabel("x")
plt.ylabel("f(x)")
plt.title("Shots")
plt.show()