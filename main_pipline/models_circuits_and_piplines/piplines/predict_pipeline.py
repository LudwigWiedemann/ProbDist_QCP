# Basic tensorflow optimisation, needs to be before every other import
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


import time
import numpy as np

from main_pipline.models_circuits_and_piplines.piplines.predict_pipline_div.predict_plots_and_metrics import \
    show_all_evaluation_plots, plot_full_timeframe_data
from main_pipline.input.div.dataset_manager import generate_time_series_data
from main_pipline.models_circuits_and_piplines.piplines.predict_pipline_div.predict_iterative_forecasting import \
    iterative_forecast

# Current list of circuits
from main_pipline.models_circuits_and_piplines.circuits.circuits import RY_Circuit, RYXZ_Circuit

# Current main model ()
from main_pipline.models_circuits_and_piplines.models.predict_variable_circuit_model import PVCModel

# Baseline model
from main_pipline.models_circuits_and_piplines.models.baseline_models.predict_hybrid_model import PHModel

# Perhaps TODO remove local config if config files are implemented or hold as alternative
full_config = {
    # training data parameter
    'time_frame_start': -4*np.pi,  # start of timeframe
    'time_frame_end': 12*np.pi,  # end of timeframe, needs to be bigger than time_frame_start
    'n_steps': 200,  # How many points are in the full timeframe
    'time_steps': 50,  # How many consecutive points are in train/test sample
    'future_steps': 10,  # How many points are predicted in train/test sample
    'num_samples': 1000,  # How many samples of time_steps/future_steps are generated from the timeframe
    'noise_level': 0.1,  # Noise level on Inputs
    'train_test_ratio': 0.6,  # The higher the ratio to more data is used for training
    # Run parameter
    'model': 'Hybrid',  # PCV is the current main_model others are for baseline
    'custom_circuit': False,  # For now only relevant for PCVModel
    'circuit': 'RY_Circuit',
    'epochs': 250,  # Adjusted to start with a reasonable number
    'batch_size': 64,  # Keep this value for now
    # Optimization parameter
    'learning_rate': 0.004,  # Adjusted to a common starting point
    'loss_function': 'mse',  # currently at 'mse'
    # Forecasting parameter
    'steps_to_predict': 300
}
# Perhaps TODO expand on models
models = {'Hybrid': PHModel, 'Variable_circuit': PVCModel}
# Perhaps TODO expand on circuits
circuits = {'RY_Circuit': RY_Circuit, 'RYXZ_Circuit': RYXZ_Circuit}

def function(x):
    return np.sin(x) + 0.5 * np.cos(2 * x) + 0.25 * np.sin(3 * x)

def run_model(dataset, config, circuits):

    # Initialised the current model
    # TODO adapt model with flag to automatically differentiate
    if config['custom_circuit']:
        #  TODO adapt circuits / model to make custom circuit choice possible
        circuit = circuits[config['circuit']]
        model = models[config['model']](circuit, config)
    else:
        model = models[config['model']](config)

    # Fit the model
    print("Starting training")
    loss_progress = model.train(dataset)
    print("Training completed")

    # Evaluate the model based on the test data
    print("Starting evaluation")
    pred_y_test_data, loss = model.evaluate(dataset)
    print(f"Test Loss: {loss}")
    pred_y_test_data, loss = model.evaluate(dataset)

    # Print plots related to the evaluation of test data
    show_all_evaluation_plots(pred_y_test_data, loss_progress, dataset, config)

    return model

def main():

    # Generate dataset
    print("Generating training data")
    dataset = generate_time_series_data(function, full_config)
    print("Training data generated")

    # Train and evaluate model and plot results
    model = run_model(dataset, full_config, circuits)

    # Forecast using the fitted model and plot results
    iterative_forecast(function, model, dataset, full_config)

if __name__ == "__main__":
    start_time = time.time()
    from silence_tensorflow import silence_tensorflow

    silence_tensorflow()

    main()
    print(f"Pipline complete in {time.time() - start_time} seconds")
