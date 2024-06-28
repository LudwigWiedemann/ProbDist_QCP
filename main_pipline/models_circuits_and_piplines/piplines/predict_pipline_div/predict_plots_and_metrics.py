# predict_plots_and_metrics.py
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np


def show_all_evaluation_plots(pred_y_test_data, loss_progress, dataset, config, logger):
    input_test = dataset['input_test']
    output_test = dataset['output_test']
    input_noisy_test = dataset['input_noisy_test']

    plot_metrics(loss_progress, show=config['show_model_plots'], logger=logger)

    n_sample_plots = int(config['num_samples'] * 0.05)
    for i in range(len(pred_y_test_data)):
        if i % n_sample_plots == 0:
            x_indices = np.arange(config['time_steps'] + config['future_steps'])
            y_real_combined = np.concatenate((input_test[i].flatten(), output_test[i].flatten()))
            y_pred_combined = np.concatenate((input_test[i].flatten(), pred_y_test_data[i].flatten()))
            plot_predictions(x_indices, input_test[i].flatten(), input_noisy_test[i].flatten(), y_real_combined,
                             y_pred_combined,
                             title=f'Test_Data_Sample_{i + 1}', show=config['show_model_plots'], logger=logger)

    plot_residuals(output_test.flatten(), pred_y_test_data.flatten(), title='Residuals on Test Data',
                   show=config['show_model_plots'], logger=logger)


def show_all_forecasting_plots(target_function, pred_y_forecast_data, dataset, config, logger):
    input_forecast = dataset['input_forecast']
    input_noisy_forecast = dataset['input_noisy_forecast']
    step_size = dataset['step_size']

    future_frame_end = step_size * config['steps_to_predict']
    real_future_values = target_function(
        np.linspace(config['time_frame_end'], config['time_frame_end'] + future_frame_end, config['steps_to_predict']))

    x_iter_indices = np.arange(config['time_steps'] + config['steps_to_predict'])
    y_iter_combined = np.concatenate((input_forecast.flatten(), real_future_values))
    plot_predictions(x_iter_indices, input_forecast.flatten(), input_noisy_forecast.flatten(), y_iter_combined,
                     np.concatenate((input_forecast.flatten(), pred_y_forecast_data)),
                     title='Iterative_Forecast', show=config['show_forecast_plots'], logger=logger)


def show_sample_preview_plots(input_test, output_test, input_noisy_test, config, logger):
    x_indices = np.arange(config['time_steps'] + config['future_steps'])
    y_real_combined = np.concatenate((input_test.flatten(), output_test.flatten()))
    plot_predictions(x_indices, input_test.flatten(), input_noisy_test.flatten(), y_real_combined, None,
                     title=f'Random_Sample_Preview', show=config['show_dataset_plots'], logger=logger)


def plot_metrics(loss_progress, show=False, logger=None):
    if isinstance(loss_progress, dict) and 'loss' in loss_progress:
        plt.figure()
        plt.plot(loss_progress['loss'], label='Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        plt.savefig(Path(logger.folder_path) / "plot_metrics.png")
        if show:
            plt.show()
    else:
        logger.info("History object does not contain 'loss'.")


def plot_predictions(x_data, input_real, input_noisy, y_real, y_pred=None, title='Real vs Predicted', show=False,
                     logger=None):
    plt.figure(figsize=(max(10, len(x_data) / 10), 10))
    plt.plot(x_data[:len(input_real)], input_real, label='Known', color='blue', marker='o', linestyle='-')
    plt.plot(x_data[:len(input_real)], input_noisy, label='Known (Noisy)', color='cyan', marker='o', linestyle='--')
    plt.plot([x_data[len(input_real) - 1], x_data[len(input_real)]], [input_real[-1], y_real[len(input_real)]],
             color='blue', linestyle='-')
    if len(y_real) > len(input_real):
        plt.plot(x_data[len(input_real):len(input_real) + len(y_real) - len(input_real)], y_real[len(input_real):],
                 label='Real Future', color='green', marker='o', linestyle='-')
    if y_pred is not None and len(y_pred) > len(input_real):
        plt.plot(x_data[len(input_real):len(input_real) + len(y_pred) - len(input_real)], y_pred[len(input_real):],
                 label='Predicted Future', color='red', marker='x', linestyle='-')
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.title(title)
    plt.legend()
    if show:
        plt.show()
    if logger.folder_path:
        plt.savefig(Path(logger.folder_path) / f"{title}_plot_predictions.png")
    else:
        print(f"Warning: Logger folder path is None. Plot {title} not saved.")
    plt.close()


def plot_residuals(y_real, y_pred, title='Residuals', show=False, logger=None):
    residuals = y_real - y_pred
    plt.figure()
    plt.plot(residuals, label='Residuals', color='purple')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Samples')
    plt.ylabel('Residuals')
    plt.title(title)
    plt.legend()
    if logger.folder_path:
        plt.savefig(Path(logger.folder_path) / f"{title}_plot_predictions.png")
    else:
        print(f"Warning: Logger folder path is None. Plot {title} not saved.")

    if show:
        plt.show()
    plt.close()


def plot_full_timeframe_data(x_data, y_data, y_noisy_data, title='Full Timeframe Data', show=False, logger=None):
    plt.figure(figsize=(max(10, len(x_data) / 10), 10))
    plt.plot(x_data, y_noisy_data, label='Noisy', color='blue', marker='x', linestyle='--')
    plt.plot(x_data, y_data, label='Real', color='cyan', marker='o', linestyle='-')
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.title(title)
    plt.legend()

    if logger is not None:
        save_path = Path(logger.folder_path) / "plot_full_timeframe_data.png"
        plt.savefig(save_path)
        logger.info(f"Plot saved to {save_path}")
    else:
        plt.savefig("plot_full_timeframe_data.png")

    if show:
        plt.show()
    plt.close()
