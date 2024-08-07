import matplotlib.pyplot as plt
from pennylane import numpy as np


def plot(data, x_start, step_size, original_data_length):
    x_axis = np.linspace(x_start, x_start + (step_size * len(data)), len(data))
    plt.plot(x_axis[0:original_data_length], data[0:original_data_length], label='known data', alpha=0.5, marker='o',
             color='blue')
    plt.plot(x_axis[original_data_length:len(x_axis)], data[original_data_length:len(data)], label='prediction',
             alpha=0.5, marker='o', color='red')
    plt.ylim(-3, 3)
    plt.grid(True)
    plt.legend()
    plt.xlabel("time steps")
    plt.ylabel("observed value")
    # plt.title("RY RZ CRY")
    plt.show()


def plot_evaluation(predictions, x_start, step_size, original_data_length):
    x_axis = np.linspace(x_start, x_start + (step_size * len(predictions[0])), len(predictions[0]))
    plt.plot(x_axis[0:original_data_length], predictions[0][0:original_data_length], label='known data', alpha=0.5, marker='o',
             color='blue')
    # Plot extended_data
    # extended_x_axis = np.linspace(0, len(predictions[0])*step_size, len(extended_data))
    # plt.plot(extended_x_axis, extended_data, label='extended data', alpha=0.2, color='green')



    for i in range(len(predictions) - 1):
        plt.plot(x_axis[original_data_length:len(x_axis)], predictions[i][original_data_length:len(predictions[i])],
                 alpha=0.05, marker='o', color='red')
    i = len(predictions) - 1
    plt.plot(x_axis[original_data_length:len(x_axis)], predictions[i][original_data_length:len(predictions[i])],
             alpha=0.05, marker='o', color='red', label='predictions')

    plt.ylim(-3, 3)
    plt.grid(True)
    plt.legend()
    plt.title('Prediction density per timestep')
    plt.xlabel("x")
    plt.ylabel("f(x)")
    # plt.title("RY RZ CRY")
    plt.show()

def plot_kl_divergence(value_list, x_start, step_size, y_label, color="red"):
    # Calculate the KL divergence
    x_axis = np.linspace(x_start, x_start + (step_size * len(value_list)), len(value_list))

    # Create a new figure
    plt.figure()

    # Plot the KL divergence
    plt.plot(x_axis, value_list, label='KL Divergence', color=color)
    plt.scatter(x_axis, value_list, color=color)

    # Set the title and labels
    plt.title('Kullback-Leibler Divergence')
    plt.xlabel('Index')
    plt.ylabel(y_label)

    # Adjust y-axis limits to fit the range of KL divergence values
    plt.ylim(0.95 * min(value_list), 1.05 * max(value_list))

    # Show the plot
    plt.show()


def f(x):
    #return np.sin(x)
    return np.sin(x) + 0.5 * np.cos(2 * x) + 0.25 * np.sin(3 * x)


def plot_sample(sample, step_size, num):
    total_len = len(sample[0]) + len(sample[1])
    x_axis = np.linspace(0, total_len * step_size, total_len)
    plt.figure()
    exp_out = [sample[0][len(sample[0]) - 1]]
    for s in sample[1]:
        exp_out.append(s)
    plt.plot(x_axis[len(sample[0])-1: total_len], exp_out, label='ground truth', marker='o', color='red')
    plt.plot(x_axis[0: len(sample[0])], sample[0], label='inputs', marker='o', color='blue')
    plt.title('Sample ' + str(num))
    plt.xlabel('time steps')
    plt.ylabel('observed value')
    plt.ylim(-3, 3)

    plt.grid(True)
    plt.legend()
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