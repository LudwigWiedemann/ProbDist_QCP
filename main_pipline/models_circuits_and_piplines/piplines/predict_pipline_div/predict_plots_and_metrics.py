import time
from pennylane import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


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
            time.sleep(3)

    plot_residuals(output_test.flatten(), pred_y_test_data.flatten(), title='Residuals on Test Data',
                   show=config['show_model_plots'], logger=logger)


def show_all_forecasting_plots(pred_y_forecast_data, dataset, config, extention, logger):
    input_forecast = dataset['input_forecast']
    input_noisy_forecast = dataset['input_noisy_forecast']
    extended_y_data = dataset['extended_y_data']

    x_iter_indices = np.arange(config['time_steps'] + config['steps_to_predict'])
    y_iter_combined = np.concatenate((input_forecast.flatten(), extended_y_data))
    plot_predictions(x_iter_indices, input_forecast.flatten(), input_noisy_forecast.flatten(), y_iter_combined,
                     np.concatenate((input_forecast.flatten(), pred_y_forecast_data)),
                     title=f'{extention}_Iterative_Forecast', show=config['show_forecast_plots'], logger=logger)


def show_all_seed_evaluation_plots(forecast_data_set, dataset, config, logger, title=''):
    input_forecast = dataset['input_forecast']
    input_noisy_forecast = dataset['input_noisy_forecast']
    extended_y_data = dataset['extended_y_data']

    x_iter_indices = np.arange(config['time_steps'] + config['steps_to_predict'])
    y_iter_combined = np.concatenate((input_forecast.flatten(), extended_y_data))
    plot_seed_predictions(x_iter_indices, input_forecast.flatten(), input_noisy_forecast.flatten(), y_iter_combined,
                     forecast_data_set,
                     title=f'5_Seeds_{title}Iterative_Forecast', show=config['show_forecast_plots'], logger=logger)
    reshaped_predictions = []
    for i in range(config['steps_to_predict']):
        step_predictions = []
        for j in range(5):
            step_predictions.append(forecast_data_set[j][i])
        reshaped_predictions.append(step_predictions)
    plot_approx_predictions(x_iter_indices, input_forecast.flatten(), input_noisy_forecast.flatten(), y_iter_combined,
                            forecast_data_set,
                            title=f'5_Seeds_{title}Iterative_Forecast', show=config['show_approx_plots'], logger=logger)
    time.sleep(3)
    plot_approx_predictions_mean(x_iter_indices, input_forecast.flatten(), input_noisy_forecast.flatten(),
                                 y_iter_combined, reshaped_predictions,
                                 title=f'5_Seeds_{title}Iterative_Forecast_mean', show=config['show_approx_plots'], logger=logger)
    time.sleep(3)
    plot_approx_predictions_box(x_iter_indices, input_forecast.flatten(), input_noisy_forecast.flatten(),
                                y_iter_combined, reshaped_predictions,
                                title=f'5_Seeds_{title}Iterative_Forecast_box', show=config['show_approx_plots'], logger=logger)
    time.sleep(3)



def show_sample_preview_plots(input_test, output_test, input_noisy_test, config, logger):
    x_indices = np.arange(config['time_steps'] + config['future_steps'])
    y_real_combined = np.concatenate((input_test.flatten(), output_test.flatten()))
    plot_predictions(x_indices, input_test.flatten(), input_noisy_test.flatten(), y_real_combined, None,
                     title=f'Random_Sample_Preview', show=config['show_dataset_plots'], logger=logger)


def show_approx_sample_plots(approx_sets, sample_index, dataset, config, logger, title='Approx_sample'):
    for i, s in enumerate(sample_index):
        approx_outputs = approx_sets[i]
        input_sample = dataset['input_test'][s]
        input_noisy_sample = dataset['input_noisy_test'][s]
        output_sample = dataset['output_test'][s]

        x_indices = np.arange(config['time_steps'] + config['future_steps'])
        y_real_combined = np.concatenate((input_sample.flatten(), output_sample.flatten()))

        reshaped_predictions = []
        for i in range(config['future_steps']):
            step_predictions = []
            for j in range(config['shot_predictions']):
                step_predictions.append(approx_outputs[j][i])
            reshaped_predictions.append(step_predictions)
        plot_approx_predictions(x_indices, input_sample.flatten(), input_noisy_sample.flatten(), y_real_combined,
                                approx_outputs,
                                title=f'{title}_{i}', show=config['show_approx_plots'], logger=logger)
        time.sleep(5)
        plot_approx_predictions_mean(x_indices, input_sample.flatten(), input_noisy_sample.flatten(), y_real_combined,
                                     np.array(reshaped_predictions),
                                     title=f'{title}_mean_{i}', show=config['show_approx_plots'], logger=logger)
        time.sleep(5)
        plot_approx_predictions_box(x_indices, input_sample.flatten(), input_noisy_sample.flatten(), y_real_combined,
                                    np.array(reshaped_predictions),
                                    title=f'{title}_box_{i}', show=config['show_approx_plots'], logger=logger)
        time.sleep(5)


def show_all_shot_forecasting_plots(pred_y_forecast_data, dataset, config, logger,
                                    title='Iterative_Forecast'):
    input_forecast = dataset['input_forecast']
    input_noisy_forecast = dataset['input_noisy_forecast']
    extended_y_data = dataset['extended_y_data']

    x_iter_indices = np.arange(config['time_steps'] + config['steps_to_predict'])
    y_iter_combined = np.concatenate((input_forecast.flatten(), extended_y_data))
    reshaped_predictions = []
    for i in range(config['steps_to_predict']):
        step_predictions = []
        for j in range(config['shot_predictions']):
            step_predictions.append(pred_y_forecast_data[j][i])
        reshaped_predictions.append(step_predictions)
    plot_approx_predictions(x_iter_indices, input_forecast.flatten(), input_noisy_forecast.flatten(), y_iter_combined,
                            pred_y_forecast_data,
                            title=f'{title}', show=config['show_approx_plots'], logger=logger)
    time.sleep(3)
    plot_approx_predictions_mean(x_iter_indices, input_forecast.flatten(), input_noisy_forecast.flatten(),
                                 y_iter_combined, reshaped_predictions,
                                 title=f'{title}_mean', show=config['show_approx_plots'], logger=logger)
    time.sleep(3)
    plot_approx_predictions_box(x_iter_indices, input_forecast.flatten(), input_noisy_forecast.flatten(),
                                y_iter_combined, reshaped_predictions,
                                title=f'{title}_box', show=config['show_approx_plots'], logger=logger)
    time.sleep(3)


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
        plt.close()
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
    if logger.folder_path:
        plt.savefig(Path(logger.folder_path) / f"{title}_plot_predictions.png")
    else:
        print(f"Warning: Logger folder path is None. Plot {title} not saved.")
    if show:
        plt.show()
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


def plot_approx_predictions(x_data, input_real, input_noisy, y_real, approx_outputs, title='Real vs Predicted',
                            show=False, logger=None):
    plt.figure(figsize=(max(10, len(x_data) / 10), 10))
    plt.plot(x_data[:len(input_real)], input_real, label='Known', color='blue', marker='o', linestyle='-')
    plt.plot(x_data[:len(input_real)], input_noisy, label='Known (Noisy)', color='cyan', marker='o', linestyle='--')
    plt.plot([x_data[len(input_real) - 1], x_data[len(input_real)]], [input_real[-1], y_real[len(input_real)]],
             color='blue', linestyle='-')

    plt.plot(x_data[len(input_real):len(y_real)], y_real[len(input_real):],
             label='Real Future', color='green', marker='o', linestyle='-')

    for output in approx_outputs:
        y_pred_combined = np.concatenate((input_real.flatten(), np.array(output).flatten()))
        plt.plot(x_data[len(input_real):len(y_pred_combined)],
                 y_pred_combined[len(input_real):],
                 marker='x', linestyle='-', alpha=0.1, color='red')
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.title(title)
    plt.legend()
    if logger.folder_path:
        plt.savefig(Path(logger.folder_path) / f"{title}_approx_prediction_.png")
    else:
        print(f"Warning: Logger folder path is None. Plot {title} not saved.")
    if show:
        plt.show()
    plt.close()


def plot_approx_predictions_mean(x_data, input_real, input_noisy, y_real, approx_outputs, title='Real vs Predicted',
                                 show=False, logger=None):
    plt.figure(figsize=(max(10, len(x_data) / 10), 10))

    # Plot the known and noisy data
    plt.plot(x_data[:len(input_real)], input_real, label='Known', color='blue', marker='o', linestyle='-')
    plt.plot(x_data[:len(input_real)], input_noisy, label='Known (Noisy)', color='cyan', marker='o', linestyle='--')
    plt.plot([x_data[len(input_real) - 1], x_data[len(input_real)]], [input_real[-1], y_real[len(input_real)]],
             color='blue', linestyle='-')

    if len(y_real) > len(input_real):
        plt.plot(x_data[len(input_real):len(y_real)], y_real[len(input_real):],
                 label='Real Future', color='green', marker='o', linestyle='-')

    # Calculate mean and standard deviation of predictions
    mean_pred = np.mean(approx_outputs, axis=1)
    std_pred = np.std(approx_outputs, axis=1)

    # Plot mean prediction with confidence interval
    y_pred_combined = np.concatenate((input_real.flatten(), mean_pred.flatten()))
    plt.plot(x_data[len(input_real):len(y_pred_combined)],
             y_pred_combined[len(input_real):], label='Mean Prediction', color='red', linestyle='-')

    plt.fill_between(x_data[len(input_real):len(input_real) + len(mean_pred)],
                     mean_pred - std_pred, mean_pred + std_pred, color='red', alpha=0.3, label='Std Dev')

    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.title(title)
    plt.legend()
    if logger.folder_path:
        plt.savefig(Path(logger.folder_path) / f"{title}_approx_prediction.png")
    else:
        print(f"Warning: Logger folder path is None. Plot {title} not saved.")
    if show:
        plt.show()
    plt.close()


def plot_approx_predictions_box(x_data, input_real, input_noisy, y_real, approx_outputs, title='Real vs Predicted',
                                show=False, logger=None):
    plt.figure(figsize=(max(10, len(x_data) / 10), 10))

    # Plot known data
    plt.plot(x_data[:len(input_real)], input_real, label='Known', color='blue', marker='o', linestyle='-')
    plt.plot(x_data[:len(input_real)], input_noisy, label='Known (Noisy)', color='cyan', marker='o', linestyle='--')
    plt.plot([x_data[len(input_real) - 1], x_data[len(input_real)]], [input_real[-1], y_real[len(input_real)]],
             color='blue', linestyle='-')

    # Plot real future data
    if len(y_real) > len(input_real):
        plt.plot(x_data[len(input_real):len(input_real) + len(y_real) - len(input_real)], y_real[len(input_real):],
                 label='Real Future', color='green', marker='o', linestyle='-')

    # Prepare boxplot data
    predictions_at_steps = [np.array(output).flatten() for output in approx_outputs]
    boxplot_positions = np.arange(len(input_real), len(input_real) + len(predictions_at_steps))

    # Truncate if necessary to match lengths
    min_length = min(len(predictions_at_steps), len(boxplot_positions))
    predictions_at_steps = predictions_at_steps[:min_length]
    boxplot_positions = boxplot_positions[:min_length]

    # Plot boxplots
    plt.boxplot(predictions_at_steps, positions=boxplot_positions, widths=0.5, patch_artist=True,
                boxprops=dict(facecolor='orange', alpha=0.3))

    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.title(title)
    plt.legend()

    if logger and logger.folder_path:
        plt.savefig(Path(logger.folder_path) / f"{title}_boxplot_prediction.png")
    else:
        print(f"Warning: Logger folder path is None. Plot {title} not saved.")

    if show:
        plt.show()

    plt.close()


def plot_seed_predictions(x_data, input_real, input_noisy, y_real, forecast_data_set, title='Real vs Predicted', show=False, logger=None):
    plt.figure(figsize=(max(10, len(x_data) / 10), 10))
    plt.plot(x_data[:len(input_real)], input_real, label='Known', color='blue', marker='o', linestyle='-')
    plt.plot(x_data[:len(input_real)], input_noisy, label='Known (Noisy)', color='cyan', marker='o', linestyle='--')
    plt.plot([x_data[len(input_real) - 1], x_data[len(input_real)]], [input_real[-1], y_real[len(input_real)]],
             color='blue', linestyle='-')
    if len(y_real) > len(input_real):
        plt.plot(x_data[len(input_real):len(input_real) + len(y_real) - len(input_real)], y_real[len(input_real):],
                 label='Real Future', color='green', marker='o', linestyle='-')

    for i in range(len(forecast_data_set)):
        y_pred = np.concatenate((input_real.flatten(), np.array(forecast_data_set[i]).flatten()))
        plt.plot(x_data[len(input_real):len(y_pred)], y_pred[len(input_real):],
                 label=f'Predicted_Future_Seed_{i}', marker='x', linestyle='-')

    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.title(title)
    plt.legend()
    if logger.folder_path:
        plt.savefig(Path(logger.folder_path) / f"{title}_plot_predictions.png")
    else:
        print(f"Warning: Logger folder path is None. Plot {title} not saved.")
    if show:
        plt.show()
    plt.close()

