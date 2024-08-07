from pennylane import numpy as np

from main_pipline.models_circuits_and_piplines.piplines.predict_pipline_div.predict_plots_and_metrics import \
    show_approx_sample_plots, show_all_shot_forecasting_plots


def evaluate_sample_with_shot(model, dataset, sample_index, config, logger, custome_shots=None, title=''):
    samples = dataset['input_test'][sample_index]

    approx_sets = []
    for sample in samples:
        predictions = []
        for _ in range(config['shot_predictions']):
            retry_attempts = 5
            while retry_attempts > 0:
                try:
                    prediction = np.array(
                        model.predict_shots(sample.reshape(config['time_steps'], ), shots=custome_shots))
                    predictions.append(prediction)
                    break
                except ValueError as e:
                    logger.error(f"ValueError: {e}, retrying...")
                    retry_attempts -= 1
                    if retry_attempts == 0:
                        logger.error(f"Failed after {5} retries.")
                        raise e
        approx_sets.append(predictions)
    show_approx_sample_plots(approx_sets, sample_index, dataset, config, logger, title=f'Approx_sample_{title}')


def fully_iterative_shot_forecast(model, dataset, config, custome_shots=None):
    input_pred = dataset['input_forecast']
    fully_predicted_ar = []
    for _ in range(config['shot_predictions']):
        output_ar = []
        for i in range(config['steps_to_predict'] // config['future_steps']):
            output_pred = model.predict_shots(input_pred.reshape(config['time_steps'], ), shots=custome_shots)
            output_ar.append(np.array(output_pred))
            input_pred = np.concatenate([np.array(input_pred).flatten(), np.array(output_pred).flatten()])[
                         -config['time_steps']:]

        fully_predicted_ar.append(np.concatenate(output_ar))
    return fully_predicted_ar

def partial_iterative_shot_forecast(model, dataset, config, logger=None, custome_shots=None, title=''):
    input_partial_pred = dataset['extended_forecast_sample'][0]
    partial_predicted_ar = []

    for _ in range(config['shot_predictions']):
        output_ar = []
        for i in range(1, len(dataset['extended_forecast_sample'])):
            output = model.predict_shots(input_partial_pred.reshape(config['time_steps'], ), shots=custome_shots)
            output_ar.append(np.array(output))
            input_partial_pred = dataset['extended_forecast_sample'][i]
        partial_predicted_ar.append(np.concatenate(output_ar))

    return partial_predicted_ar