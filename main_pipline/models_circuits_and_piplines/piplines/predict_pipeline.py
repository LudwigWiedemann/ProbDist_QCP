import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import time
from silence_tensorflow import silence_tensorflow
import numpy as np
from main_pipline.input.div.logger import logger
import main_pipline.input.div.filemanager as filemanager

from main_pipline.models_circuits_and_piplines.piplines.predict_pipline_div.predict_plots_and_metrics import show_all_evaluation_plots
from main_pipline.input.div.dataset_manager import generate_time_series_data
from main_pipline.models_circuits_and_piplines.piplines.predict_pipline_div.predict_iterative_forecasting import iterative_forecast

from main_pipline.models_circuits_and_piplines.circuits.variable_circuit import new_RYXZ_Circuit, new_baseline
from main_pipline.models_circuits_and_piplines.circuits.amp_Circuit import base_Amp_Circuit, layered_Amp_Circuit, \
    tangle_Amp_Circuit, test_Amp_Circuit, double_Amp_Circuit

from main_pipline.models_circuits_and_piplines.models.predict_variable_circuit_model import PVCModel
from main_pipline.models_circuits_and_piplines.models.predict_amp_circuit_model import PACModel

from main_pipline.models_circuits_and_piplines.models.baseline_models.predict_hybrid_model import PHModel

full_config = {
    # Dataset parameter
    'time_frame_start': -4 * np.pi,
    'time_frame_end': 12 * np.pi,
    'n_steps': 80,
    'time_steps': 8,
    'future_steps': 4,
    'num_samples': 80,
    'noise_level': 0.1,
    'train_test_ratio': 0.6,
    #  Plotting parameter
    'preview_samples': 3,
    'show_dataset_plots': True,
    'show_model_plots': False,
    'show_forecast_plots': True,
    'steps_to_predict': 300,
    # Model parameter
    'model': 'Amp_circuit',
    'circuit': 'Tangle_Amp_Circuit',
    # Run parameter
    'epochs': 1,
    'batch_size': 32,
    'learning_rate': 0.5,
    'loss_function': 'mse',
    'compress_factor': 1.5,
    'patience': 10,
    'min_delta': 0.001,
    # Circuit parameter
    'layers': 2,  # Only Optuna/Tangle circuit
    'shots': None
}

models = {'Hybrid': PHModel, 'Variable_circuit': PVCModel, 'Amp_circuit': PACModel}
circuits = {'new_RYXZ_Circuit': new_RYXZ_Circuit, 'new_baseline': new_baseline, 'base_Amp_Circuit': base_Amp_Circuit,
            'layered_Amp_Circuit': layered_Amp_Circuit,'Tangle_Amp_Circuit':tangle_Amp_Circuit, 'Test_Circuit': test_Amp_Circuit,
            'Double_Amp_Circuit': double_Amp_Circuit}

def function(x):
    return np.sin(x) + 0.5 * np.cos(2 * x) + 0.25 * np.sin(3 * x)

def run_model(dataset, config):
    circuit = circuits[config['circuit']]
    model = models[config['model']](circuit, config)
    logger.info(f"Model: {config['model']}, Circuit: {config['circuit']}")
    logger.info("Config of this run:")
    for key in config.keys():
        logger.info(f"{key} : {config[key]}")
    model_save_path = os.path.join(filemanager.path, "fitted_model")

    logger.info("Starting training")
    loss_progress = model.train(dataset)
    logger.info("Training completed")

    model.save_model(model_save_path)

    logger.info("Starting evaluation")
    pred_y_test_data, loss = model.evaluate(dataset)
    logger.info(f"Test Loss: {loss}")

    show_all_evaluation_plots(pred_y_test_data, loss_progress, dataset, config)

    return model, loss

def main():
    logger.info("Generating training data")
    dataset = generate_time_series_data(function, full_config)
    logger.info("Training data generated")

    model, loss = run_model(dataset, full_config)

    iterative_forecast(function, model, dataset, full_config)

if __name__ == "__main__":
    start_time = time.time()
    silence_tensorflow()
    main()
    logger.info(f"Pipeline complete in {time.time() - start_time} seconds")
