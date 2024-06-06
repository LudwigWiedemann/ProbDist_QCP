from time import sleep
import matplotlib.pyplot as plt
import numpy as np


# TODO save plots and raw metrics as files in output folder
# Perhaps TODO add additional plots
def show_all_evaluation_plots(pred_y_test_data, loss_progress, dataset, config):
    # Extract values out of dataset
    input_test = dataset['input_test']
    output_test = dataset['output_test']

    # Plot training metrics
    plot_metrics(loss_progress)

    # Plot predictions vs real values for each test sample
    for i in range(len(pred_y_test_data)):
        if i % config['num_samples'] * 0.25 == 0:
            x_indices = np.arange(config['time_steps'] + config['future_steps'])
            y_real_combined = np.concatenate((input_test[i].flatten(), output_test[i].flatten()))
            y_pred_combined = np.concatenate((input_test[i].flatten(), pred_y_test_data[i].flatten()))
            plot_predictions(x_indices, input_test[i].flatten(), y_real_combined, y_pred_combined,
                             noise_level=config['noise_level'],
                             title=f'Test Data Sample {i + 1}: Real vs Predicted')
            sleep(1.5)

    # Plot residuals for test data
    plot_residuals(output_test.flatten(), pred_y_test_data.flatten(), title='Residuals on Test Data')


def show_all_forecasting_plots(target_function, pred_y_forecast_data, dataset, config):
    input_forecast = dataset['input_forecast']
    # Calculate all real future values at once
    real_future_values = target_function(
        np.linspace(config['time_frame_end'], config['time_frame_end'] + config['steps_to_predict'],
                    config['steps_to_predict']))

    # Generate x-axes for iterative predictions
    x_iter_indices = np.arange(config['time_steps'] + config['steps_to_predict'])
    y_iter_combined = np.concatenate((input_forecast.flatten(), real_future_values))
    plot_predictions(x_iter_indices, input_forecast.flatten(), y_iter_combined,
                     np.concatenate((input_forecast.flatten(), pred_y_forecast_data)),
                     noise_level=config['noise_level'],
                     title='Iterative Forecast: Real vs Predicted', marker_distance=5)


def plot_metrics(loss_progress):
    # Check if history is a dictionary and contains 'loss' key
    if isinstance(loss_progress, dict) and 'loss' in loss_progress:
        plt.figure()
        plt.plot(loss_progress['loss'], label='Training Loss')
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

    # Connect the last known point to the first predicted point
    if len(y_pred) > len(y_known):  # Only connect if there's a future prediction
        plt.plot([x_data[len(y_known) - 1], x_data[len(y_known)]], [y_known[-1], y_real[len(y_known)]], color='blue',
                 linestyle='-')

    # Plot the real future steps with line and markers
    if len(y_real) > len(y_known):  # Check if y_real has more points than y_known
        plt.plot(x_data[len(y_known):len(y_real)], y_real[len(y_known):], label='Real Future', color='green',
                 marker='o', linestyle='-')

    # Plot the predicted future steps with line and markers
    if len(y_pred) > len(y_known):  # Check if y_pred has more points than y_known
        plt.plot(x_data[len(y_known):len(y_pred)], y_pred[len(y_known):], label='Predicted Future', color='red',
                 marker='x', linestyle='-')

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

def plot_full_timeframe_data(x_data, y_data, title='Full Timeframe Data', marker_distance=5):
    plt.figure()
    plt.plot(x_data, y_data, label='Data', color='blue', marker='o', linestyle='-')

    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.title(title)
    plt.legend()
    plt.show()
