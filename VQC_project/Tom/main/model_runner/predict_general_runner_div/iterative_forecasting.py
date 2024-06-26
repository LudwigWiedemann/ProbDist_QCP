import numpy as np

from main_pipline.models_circuits_and_piplines.piplines.predict_pipline_div.predict_plots_and_metrics import \
    show_all_forecasting_plots


def iterative_forecast(function, model, dataset, config):
    # Samples used for forecasting

    steps = config['steps_to_predict']
    time_steps = config['time_steps']
    future_steps = config['future_steps']

    input_forecast = dataset['Input_forecast']
    # Start of prediction
    current_input = input_forecast
    all_predictions = []

    for _ in range(steps // future_steps):
        # Ensure the current input has the correct shape
        if current_input.shape[1] < time_steps:
            padding = np.zeros((current_input.shape[0], time_steps - current_input.shape[1], current_input.shape[2]))
            current_input = np.concatenate((padding, current_input), axis=1)

        pred = model.predict(current_input)
        all_predictions.append(pred.flatten())
        # Use the last `time_steps` of the combined current_input + prediction as the next input
        current_input = np.concatenate((current_input.flatten(), pred.flatten()))[-time_steps:].reshape(1, -1, 1)
    show_all_forecasting_plots(function, input_forecast, np.concatenate(all_predictions), config)

