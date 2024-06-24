from time import sleep
import matplotlib.pyplot as plt
from main_pipline.input.div.logger import logger
import main_pipline.input.div.filemanager as file
import numpy as np


def show_all_evaluation_plots(pred_y_test_data, loss_progress, dataset, config):
    # Extract values out of dataset
    input_test = dataset['input_test']
    output_test = dataset['output_test']
    input_noisy_test = dataset['input_noisy_test']

    # Plot training metrics
    plot_metrics(loss_progress)

    # Plot predictions vs real values for each test sample
    n_sample_plots = int(config['num_samples'] * 0.05)
    for i in range(len(pred_y_test_data)):
        if i % n_sample_plots == 0:
            x_indices = np.arange(config['time_steps'] + config['future_steps'])
            y_real_combined = np.concatenate((input_test[i].flatten(), output_test[i].flatten()))
            y_pred_combined = np.concatenate((input_test[i].flatten(), pred_y_test_data[i].flatten()))
            plot_predictions(x_indices, input_test[i].flatten(), input_noisy_test[i].flatten(), y_real_combined,
                             y_pred_combined,
                             title=f'Test Data Sample {i + 1}: Real vs Predicted')
            sleep(1.5)

    # Plot residuals for test data
    plot_residuals(output_test.flatten(), pred_y_test_data.flatten(), title='Residuals on Test Data')


def show_all_forecasting_plots(target_function, pred_y_forecast_data, dataset, config):
    input_forecast = dataset['input_forecast']
    input_noisy_forecast = dataset['input_noisy_forecast']
    step_size = dataset['step_size']

    # Calculate all real future values at once
    future_frame_end = step_size * config['steps_to_predict']
    real_future_values = target_function(
        np.linspace(config['time_frame_end'], config['time_frame_end'] + future_frame_end, config['steps_to_predict']))

    # Generate x-axes for iterative predictions
    x_iter_indices = np.arange(config['time_steps'] + config['steps_to_predict'])
    y_iter_combined = np.concatenate((input_forecast.flatten(), real_future_values))
    plot_predictions(x_iter_indices, input_forecast.flatten(), input_noisy_forecast.flatten(), y_iter_combined,
                     np.concatenate((input_forecast.flatten(), pred_y_forecast_data)),
                     title='Iterative Forecast: Real vs Predicted', )


def show_sample_preview_plots(input_test, output_test, input_noisy_test, config):
    x_indices = np.arange(config['time_steps'] + config['future_steps'])
    y_real_combined = np.concatenate((input_test.flatten(), output_test.flatten()))
    plot_predictions(x_indices, input_test.flatten(), input_noisy_test.flatten(), y_real_combined, None,
                     title=f'Random Sample Preview')


def plot_metrics(loss_progress):
    # Check if history is a dictionary and contains 'loss' key
    if isinstance(loss_progress, dict) and 'loss' in loss_progress:
        plt.figure()
        plt.plot(loss_progress['loss'], label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.savefig(file.path + "/plot_metrics.png")
        plt.show()
    else:
        logger.info("History object does not contain 'loss'.")


def plot_predictions(x_data, input_real, input_noisy, y_real, y_pred=None, title='Real vs Predicted'):
    plt.figure(figsize=(max(10, len(x_data) / 10), 10))

    # Plot the known time steps with line and markers
    plt.plot(x_data[:len(input_real)], input_real, label='Known', color='blue', marker='o', linestyle='-')

    # Plot the noise x
    plt.plot(x_data[:len(input_real)], input_noisy, label='Known (Noisy)', color='cyan', marker='o', linestyle='--')

    # Connect the last known point to the first predicted point
    plt.plot([x_data[len(input_real) - 1], x_data[len(input_real)]], [input_real[-1], y_real[len(input_real)]],
                 color='blue', linestyle='-')

    # Plot the real future steps with line and markers
    if len(y_real) > len(input_real):  # Check if y_real has more points than y_known
        plt.plot(x_data[len(input_real):len(input_real) + len(y_real) - len(input_real)], y_real[len(input_real):],
                 label='Real Future', color='green', marker='o', linestyle='-')

    # Plot the predicted future steps with line and markers
    if y_pred is not None and len(y_pred) > len(input_real):
        plt.plot(x_data[len(input_real):len(input_real) + len(y_pred) - len(input_real)], y_pred[len(input_real):],
                 label='Predicted Future', color='red', marker='x', linestyle='-')

    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.title(title)
    plt.legend()
    plt.savefig(file.path + f"/plot_predictions_{title}.png")
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
    plt.savefig(file.path + "/plot_residuals.png")
    plt.show()


def plot_full_timeframe_data(x_data, y_data, y_noisy_data, title='Full Timeframe Data'):
    plt.figure(figsize=(20, 10))

    plt.plot(x_data, y_noisy_data, label='Noisy', color='red', marker='x', linestyle='--')
    plt.plot(x_data, y_data, label='Real', color='green', marker='o', linestyle='-')

    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.title(title)
    plt.legend()
    plt.savefig(file.path + "/plot_full_timeframe_data.png")
    plt.show()
