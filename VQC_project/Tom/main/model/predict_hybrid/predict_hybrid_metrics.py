import matplotlib.pyplot as plt
import numpy as np


def plot_metrics(history):
    # Check if history is a dictionary and contains 'loss' key
    if isinstance(history, dict) and 'loss' in history:
        plt.figure()
        plt.plot(history['loss'], label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.show()
    else:
        print("History object does not contain 'loss'.")


import matplotlib.pyplot as plt
import numpy as np


def plot_metrics(history):
    # Check if history is a dictionary and contains 'loss' key
    if isinstance(history, dict) and 'loss' in history:
        plt.figure()
        plt.plot(history['loss'], label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.show()
    else:
        print("History object does not contain 'loss'.")


def plot_predictions(x_data, y_known, y_real, y_pred, noise_level=0, title='Real vs Predicted', marker_distance=5):
    plt.figure()

    # Add noise to the known data if noise level is specified
    if noise_level > 0:
        y_known_noisy = y_known + np.random.normal(0, noise_level, len(y_known))
        plt.plot(x_data[:len(y_known)], y_known_noisy, label='Known (Noisy)', color='cyan', marker='o', linestyle='--')

    # Plot the known time steps with line and markers
    plt.plot(x_data[:len(y_known)], y_known, label='Known', color='blue', marker='o', linestyle='-')

    # Plot the real future steps with line and markers
    if len(y_real) > len(y_known):  # Check if y_real has more points than y_known
        plt.plot(x_data[len(y_known):len(y_real)], y_real[len(y_known):], label='Real Future', color='green',
                 marker='o', linestyle='-')

    # Plot the predicted future steps with line and markers
    if len(y_pred) > len(y_known):  # Check if y_pred has more points than y_known
        plt.plot(x_data[len(y_known):len(y_pred)], y_pred[len(y_known):], label='Predicted Future', color='red',
                 marker='x', linestyle='-')

    # Connect the last known point to the first predicted point
    if len(y_pred) > len(y_known):  # Only connect if there's a future prediction
        plt.plot([x_data[len(y_known) - 1], x_data[len(y_known)]], [y_known[-1], y_real[len(y_known)]], color='blue',
                 linestyle='-')

    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_residuals(y_real, y_pred, title='Residuals'):
    residuals = y_real - y_pred
    plt.figure()
    plt.plot(residuals, label='Residuals', color='purple')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Samples')
    plt.ylabel('Residuals')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_residuals(y_real, y_pred, title='Residuals'):
    residuals = y_real - y_pred
    plt.figure()
    plt.plot(residuals, label='Residuals', color='purple')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Samples')
    plt.ylabel('Residuals')
    plt.title(title)
    plt.legend()
    plt.show()
