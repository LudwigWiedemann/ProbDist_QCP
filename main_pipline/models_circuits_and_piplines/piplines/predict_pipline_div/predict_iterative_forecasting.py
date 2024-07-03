import numpy as np
from main_pipline.models_circuits_and_piplines.piplines.predict_pipline_div.predict_plots_and_metrics import \
    show_all_forecasting_plots


import numpy as np
from main_pipline.models_circuits_and_piplines.piplines.predict_pipline_div.predict_plots_and_metrics import show_all_forecasting_plots

def iterative_forecast(function, model, dataset, config, logger=None):
    steps = config['steps_to_predict']
    time_steps = config['time_steps']
    future_steps = config['future_steps']

    pred_input = dataset['input_forecast']
    real_input = dataset['input_forecast']

    all_pred_predictions = []
    all_real_predictions = []

    for i in range(steps // future_steps):
        if pred_input.shape[1] < time_steps:
            padding = np.zeros((pred_input.shape[0], time_steps - pred_input.shape[1], pred_input.shape[2]))
            pred_input = np.concatenate((padding, pred_input), axis=1)

        pred_pred = model.predict(pred_input)
        all_pred_predictions.append(pred_pred.flatten())
        pred_input = np.concatenate((pred_input.flatten(), pred_pred.flatten()))[-time_steps:].reshape(1, -1, 1)


        if real_input.shape[1] < time_steps:
            padding = np.zeros((real_input.shape[0], time_steps - real_input.shape[1], real_input.shape[2]))
            real_input = np.concatenate((padding, real_input), axis=1)

        real_pred = model.predict(real_input)
        all_real_predictions.append(real_pred.flatten())

        step_size = dataset['step_size']

        future_frame_end = step_size * config['future_steps'] * i
        real_data = function(
            np.linspace(config['time_frame_end'], config['time_frame_end'] + future_frame_end,
                        config['steps_to_predict']))

        real_input = np.concatenate((real_input.flatten(), real_data.flatten()))[-time_steps:].reshape(1, -1, 1)

    show_all_forecasting_plots(function, np.concatenate(all_pred_predictions), dataset, config, logger=logger)
    show_all_forecasting_plots(function, np.concatenate(all_real_predictions), dataset, config, logger=logger)