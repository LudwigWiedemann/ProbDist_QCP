import matplotlib.pyplot as plt
import numpy as np


def plot_metrics(history):
    # Plot the training loss
    plt.figure()
    plt.plot(history.history['loss'], label='Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.show()


def plot_predictions(x_data, y_real, y_pred, title='Real vs Predicted'):
    # Plot the real vs predicted values
    plt.figure()
    plt.plot(x_data, y_real, label='Real', color='blue')
    plt.plot(x_data, y_pred, label='Predicted', color='red')
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.title(title)
    plt.legend()
    plt.show()


# Example usage
if __name__ == "__main__":
    # Generate dummy data for testing the plot functions
    history = type('History', (object,), {'history': {'loss': [0.1, 0.08, 0.06, 0.04, 0.02]}})()
    plot_metrics(history)

    x_data = np.linspace(0, 2 * np.pi, 50).reshape(-1, 1)
    y_real = np.sin(x_data).flatten()
    y_pred = y_real + np.random.normal(0, 0.1, y_real.shape)  # Adding some noise to simulate predictions
    plot_predictions(x_data.flatten(), y_real, y_pred, title='Test Data: Real vs Predicted')
