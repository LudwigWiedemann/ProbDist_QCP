import matplotlib.pyplot as plt
from pennylane import numpy as np


def plot(data, x_start, step_size, original_data_length):
    x_axis = np.linspace(x_start, x_start + (step_size * len(data)), len(data))
    plt.plot(x_axis[0:original_data_length], data[0:original_data_length], label='known data', alpha=0.5, marker='o',
             color='blue')
    plt.plot(x_axis[original_data_length:len(x_axis)], data[original_data_length:len(data)], label='prediction',
             alpha=0.5, marker='o', color='red')
    plt.ylim(-2, 2)
    plt.grid(True)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    # plt.title("RY RZ CRY")
    plt.show()


def plot_evaluation(predictions, x_start, step_size, original_data_length):
    x_axis = np.linspace(x_start, x_start + (step_size * len(predictions[0])), len(predictions[0]))
    plt.plot(x_axis[0:original_data_length], predictions[0][0:original_data_length], label='known data', alpha=0.5, marker='o',
             color='blue')
    for i in range(len(predictions)):
        plt.plot(x_axis[original_data_length:len(x_axis)], predictions[i][original_data_length:len(predictions[i])],
                 alpha=0.01, color='red')
    plt.ylim(-2, 2)
    plt.grid(True)
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    # plt.title("RY RZ CRY")
    plt.show()


def f(x):
    #return np.sin(x)
    return np.sin(x) + 0.5 * np.cos(2 * x) + 0.25 * np.sin(3 * x)

def plot_kl_divergence(distributions):
    # Create an array of indices for the x-coordinates of the bars
    x_coords = np.arange(distributions.size)
    print("COORDINATES:")
    print(distributions)
    # Create a bar plot
    plt.bar(x_coords, distributions)
    plt.xlabel('Distribution Pair')
    plt.ylabel('Average KL Divergence')
    plt.title('Average KL Divergence for Each Pair of Distributions')
    plt.show()

#
# x_axis = np.linspace(-3, 6, 21000)
# plt.plot(x_axis, f(x_axis), label="input", alpha=0.5)
# plt.ylim(-2, 2)
# plt.grid(True)
# plt.legend()
# plt.xlabel("x")
# plt.ylabel("f(x)")
# plt.title("fhots")
# plt.show()