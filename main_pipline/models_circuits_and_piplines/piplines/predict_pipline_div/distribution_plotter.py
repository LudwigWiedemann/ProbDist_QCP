import numpy as np
import matplotlib.pyplot as plt

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

def plot_kl_divergence_dice(normalized_value_counts):
    keys = sorted(set(normalized_value_counts.keys()))
    normalized_value_counts_list = [normalized_value_counts.get(key, 0) for key in keys]
    normalized_output_counts_list = [normalized_output_counts.get(key, 0) for key in keys]

    # Plotting
    x = range(len(keys))  # X-axis points
    plt.bar(x, normalized_value_counts_list, width=0.4, label='Normalized Value Counts', align='center')
    plt.bar(x, normalized_output_counts_list, width=0.4, label='Normalized Output Counts', align='edge')

    # Adding details
    plt.xlabel('Unique Values/Bins')
    plt.ylabel('Normalized Counts')
    plt.title('Comparison of Normalized Value Counts and Normalized Output Counts')
    plt.xticks(x, keys, rotation='vertical')  # Set x-ticks to be the keys, rotate for readability
    plt.legend()

    # Display the plot
    plt.tight_layout()  # Adjust layout to not cut off labels
    plt.show()