import random

import numpy as np

from main_pipline.models_circuits_and_piplines.piplines.predict_pipline_div.predict_plots_and_metrics import \
    show_approx_sample_plots, show_all_shot_forecasting_plots


def evaluate_sample_with_shot(model, dataset, config, logger):
    sample_index = random.sample(range(len(dataset['input_test'])), config['approx_samples'])
    samples = dataset['input_test'][sample_index] / config['compress_factor']

    approx_sets = []
    for sample in samples:
        predictions = []
        for _ in range(config['shot_predictions']):
            retry_attempts = 5
            while retry_attempts > 0:
                try:
                    prediction = np.array(model.predict_shots(sample.reshape(config['time_steps'], )))
                    predictions.append(prediction * config['compress_factor'])
                    break
                except ValueError as e:
                    logger.error(f"ValueError: {e}, retrying...")
                    retry_attempts -= 1
                    if retry_attempts == 0:
                        logger.error(f"Failed after {5} retries.")
                        raise e
        approx_sets.append(predictions)
    show_approx_sample_plots(approx_sets, sample_index, dataset, config, logger)


def iterative_shot_forecast(function, model, dataset, config, logger=None):
    steps = config['steps_to_predict']
    time_steps = config['time_steps']
    future_steps = config['future_steps']

    pred_input = dataset['input_forecast']
    real_input = dataset['input_forecast']

    pred_shot_predictions = []
    real_shot_predictions = []

    for _ in range(config['shot_predictions']):
        all_pred_predictions = []
        all_real_predictions = []

        for i in range(steps // future_steps):
            if pred_input.shape[1] < time_steps:
                padding = np.zeros((pred_input.shape[0], time_steps - pred_input.shape[1], pred_input.shape[2]))
                pred_input = np.concatenate((padding, pred_input), axis=1)

            pred_pred = model.predict_shots(pred_input.reshape(config['time_steps'], ))
            all_pred_predictions.append(np.array(pred_pred).flatten())
            pred_input = np.concatenate((pred_input.flatten(), np.array(pred_pred).flatten()))[-time_steps:].reshape(1, -1, 1)

            if real_input.shape[1] < time_steps:
                padding = np.zeros((real_input.shape[0], time_steps - real_input.shape[1], real_input.shape[2]))
                real_input = np.concatenate((padding, real_input), axis=1)

            real_pred = model.predict_shots(real_input.reshape(config['time_steps'], ))
            all_real_predictions.append(np.array(real_pred).flatten())

            step_size = dataset['step_size']

            future_frame_end = step_size * config['future_steps'] * i
            real_data = function(
                np.linspace(config['time_frame_end'], config['time_frame_end'] + future_frame_end,
                            config['steps_to_predict']))

            real_input = np.concatenate((real_input.flatten(), real_data.flatten()))[-time_steps:].reshape(1, -1, 1)
        pred_shot_predictions.append(all_pred_predictions)
        real_shot_predictions.append(all_real_predictions)

    show_all_shot_forecasting_plots(function, pred_shot_predictions, dataset, config, logger=logger)
    show_all_shot_forecasting_plots(function, real_shot_predictions, dataset, config, logger=logger)
