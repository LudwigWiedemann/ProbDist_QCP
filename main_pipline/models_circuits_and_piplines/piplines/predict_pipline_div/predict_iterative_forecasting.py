import numpy as np
from main_pipline.models_circuits_and_piplines.piplines.predict_pipline_div.predict_plots_and_metrics import \
    show_all_forecasting_plots


def iterative_forecast(model, dataset, config, logger=None):
    input_pred = dataset['input_forecast'] / config['compress_factor']
    output_ar = []
    for i in range(config['steps_to_predict'] // config['future_steps']):
        output_pred = model.predict(input_pred.reshape(-1, config['time_steps'], 1))
        output_ar.append(np.array(output_pred) * config['compress_factor'])
        input_pred = np.concatenate([np.array(input_pred).flatten(), np.array(output_pred).flatten()])[
                     -config['time_steps']:]

    show_all_forecasting_plots(np.concatenate(output_ar).flatten(), dataset, config, logger=logger,
                               extention=f'Fully_Iterative_Forecast')

    input_partial_pred = dataset['extended_forecast_sample'][0] / config['compress_factor']
    output_ar = []
    for i in range(1, len(dataset['extended_forecast_sample'])):
        output = model.predict(input_partial_pred.reshape(-1, config['time_steps'], 1))
        output_ar.append(np.array(output) * config['compress_factor'])
        input_partial_pred = dataset['extended_forecast_sample'][i] / config['compress_factor']

    show_all_forecasting_plots(np.concatenate(output_ar).flatten(), dataset, config, logger=logger,
                               extention=f'Partial_Iterative_Forecast')
