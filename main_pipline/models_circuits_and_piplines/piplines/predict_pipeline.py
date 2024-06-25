# Basic tensorflow optimisation, needs to be before every other import
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import time
from silence_tensorflow import silence_tensorflow
import numpy as np
from main_pipline.input.div.logger import logger
import main_pipline.input.div.filemanager as filemanager

from main_pipline.models_circuits_and_piplines.piplines.predict_pipline_div.predict_plots_and_metrics import \
    show_all_evaluation_plots
from main_pipline.input.div.dataset_manager import generate_time_series_data
from main_pipline.models_circuits_and_piplines.piplines.predict_pipline_div.predict_iterative_forecasting import \
    iterative_forecast

# Current list of circuits
from main_pipline.models_circuits_and_piplines.circuits.variable_circuit import new_RYXZ_Circuit, new_baseline
from main_pipline.models_circuits_and_piplines.circuits.amp_Circuit import base_Amp_Circuit, layered_Amp_Circuit, \
    Tangle_Amp_Circuit, Test_Circuit

# Current main model ()
from main_pipline.models_circuits_and_piplines.models.predict_variable_circuit_model import PVCModel
from main_pipline.models_circuits_and_piplines.models.predict_amp_circuit_model import PACModel

# Baseline model
from main_pipline.models_circuits_and_piplines.models.baseline_models.predict_hybrid_model import PHModel

# Perhaps TODO remove local config if config files are implemented or hold as alternative
full_config = {
    # training data parameter
    'time_frame_start': -4 * np.pi,  # start of timeframe
    'time_frame_end': 8 * np.pi,  # end of timeframe, needs to be bigger than time_frame_start
    'n_steps': 200,  # How many points are in the full timeframe
    'time_steps': 64,  # How many consecutive points are in train/test sample
    'future_steps': 6,  # How many points are predicted in train/test sample
    'num_samples': 200,  # How many samples of time_steps/future_steps are generated from the timeframe
    'noise_level': 0.05,  # Noise level on Inputs
    'train_test_ratio': 0.6,  # The higher the ratio to more data is used for training
    'preview_samples': 1,  # How many preview samples should be included
    # Run parameter
    'model': 'Amp_circuit',  # PCV is the current main_model others are for baseline
    'circuit': 'layered_Amp_Circuit',
    'epochs': 60,  # Adjusted to start with a reasonable number
    'batch_size': 32,  # Keep this value for now
    # Optimization parameter
    'learning_rate': 0.05,  # Adjusted to a common starting point
    'loss_function': 'mse',  # currently at 'mse'
    # Forecasting parameter
    'steps_to_predict': 300

}
# Perhaps TODO expand on models
models = {'Hybrid': PHModel, 'Variable_circuit': PVCModel, 'Amp_circuit': PACModel}
# Perhaps TODO expand on circuits
circuits = {'new_RYXZ_Circuit': new_RYXZ_Circuit, 'new_baseline': new_baseline,
            'base_Amp_Circuit': base_Amp_Circuit, 'layered_Amp_Circuit': layered_Amp_Circuit,
            'Test_Circuit': Test_Circuit
            }


def function(x):
    return np.sin(x) + 0.5 * np.cos(2 * x) + 0.25 * np.sin(3 * x)


def run_model(dataset, config):
    # Initialised the current model
    circuit = circuits[config['circuit']]
    model = models[config['model']](circuit, config)

    model_save_path = os.path.join(filemanager.path, "fitted_model")

    # Fit the model
    logger.info("Starting training")
    loss_progress = model.train(dataset)
    logger.info("Training completed")

    # Save the fitted model
    #model.save_model(model_save_path)

    # Evaluate the model based on the test data
    logger.info("Starting evaluation")
    pred_y_test_data, loss = model.evaluate(dataset)
    logger.info(f"Test Loss: {loss}")

    # Print plots related to the evaluation of test data
    show_all_evaluation_plots(pred_y_test_data, loss_progress, dataset, config)

    return model


def main():
    # Generate dataset
    logger.info("Generating training data")
    dataset = generate_time_series_data(function, full_config)
    logger.info("Training data generated")

    # Train and evaluate model and plot results
    model = run_model(dataset, full_config)

    # Forecast using the fitted model and plot results
    iterative_forecast(function, model, dataset, full_config)


if __name__ == "__main__":
    # full_config = loader.dialog_load_config()
    # filemanager.create_folder()  # Creates Folder
    start_time = time.time()
    silence_tensorflow()
    main()
    logger.info(f"Pipline complete in {time.time() - start_time} seconds")
    logger.info(f"Config of this run: {full_config}")

