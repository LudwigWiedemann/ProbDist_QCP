import main_pipline.models_circuits_and_piplines.piplines.predict_pipline_div.distribution_plotter as plt
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def plot_kl_divergence(value_list, x_start, step_size, y_label, color="red", logger=None):
    """
    Plots the KL divergence values (and JS).
    :param value_list: list[list[float]]
    :param x_start: float that represents the starting point of the x-axis
    :param step_size: float that represents the step size of the x-axis
    :param y_label: str: printed y_label
    :param color: str: color of plot
    :param logger: Logger
    :return: None
    """
    # Generate x-axis values
    x_axis = np.linspace(x_start, x_start + (step_size * len(value_list)), len(value_list))

    # Create a new figure
    plt.figure()

    # Plot the divergence
    plt.plot(x_axis, value_list, color=color)
    plt.scatter(x_axis, value_list, color=color)

    # Set the labels
    plt.xlabel('Index')
    plt.ylabel(y_label)

    # Adjust y-axis limits to fit the range of KL divergence values
    plt.ylim(0.95 * min(value_list), 1.05 * max(value_list))
    if logger.folder_path:
        # Save the plot
        plt.savefig(Path(logger.folder_path) / f"{y_label}_plot_predictions.png")
        logger.info(f"Plot {y_label} saved at {Path(logger.folder_path) / f'{y_label}_plot_predictions.png'}")
        show=True
    else:
        # Display the plot if no logger is provided
        print(f"Warning: Logger folder path is None. Plot {y_label} not saved.")
        show=True
    if show:
        plt.show()

def plot_divergence_at_i(counts_input_i, counts_prediction_i, i, bits, logger):
    """
    Plots the distribution of counts at a specific bit i.
    :param counts_input_i: list[float]: normalized input counts (base)
    :param counts_prediction_i:  list[float]: normalized prediction counts
    :param i: float: x-axis value for saving the plot
    :param bits: float: kl-divergence bits for saving the plot
    :param logger: Logger
    :return: None
    """
    # Generate x-axis values starting from 1 to the length of the input lists
    x = np.arange(1, len(counts_input_i) + 1)

    # Set the width for the bars and their colors
    width = 0.35
    color_input = 'blue'  # Color for input bars
    color_prediction = 'orange'  # Color for prediction bars

    # Plotting
    bars_input = plt.bar(x - width/2, counts_input_i, width, label='Normalized Input Counts', color=color_input)
    bars_prediction = plt.bar(x + width/2, counts_prediction_i, width, label='Normalized Prediction Counts', color=color_prediction)

    # Adding the value above each bar with matching color
    for bar in bars_input:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom', color=color_input)

    for bar in bars_prediction:
        height = bar.get_height()
        if height > 0:
            plt.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.2f}', ha='center', va='bottom', color=color_prediction)

    # Adding details
    plt.xlabel('Bins')
    plt.ylabel('Normalized Counts')
    plt.xticks(x)  # Set x-ticks to be the x values, ensuring a label for every value
    plt.legend()

    # Display the plot
    plt.tight_layout()  # Adjust layout to not cut off labels
    if logger and logger.folder_path:
        # Save the plot
        plt.savefig(Path(logger.folder_path) / f"distribution_at_{i}_bits{bits}_plot_predictions.png")
        logger.info(f"Plot_divergence at {i} saved at {Path(logger.folder_path) / f'distribution_at_{i}_bits{bits}_plot_predictions.png'}")
    else:
        # Display the plot if no logger is provided
        print(f"Warning: Logger folder path is None. Plot {i} not saved.")
    plt.show()

def plot_kl_divergence_dice(normalized_value_counts, normalized_output_counts):
    """
    Plots the comparison of normalized value counts and normalized output counts.
    :param normalized_value_counts: list[float]: normalized value counts
    :param normalized_output_counts: list[float]: normalized output counts
    :return: None
    """
    # Sort the keys and get the corresponding values
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