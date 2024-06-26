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
from main_pipline.models_circuits_and_piplines.circuits.amp_Circuit import base_Amp_Circuit, layered_Amp_Circuit, Tangle_Amp_Circuit, Test_Circuit

from main_pipline.models_circuits_and_piplines.models.predict_variable_circuit_model import PVCModel
from main_pipline.models_circuits_and_piplines.models.predict_amp_circuit_model import PACModel

from main_pipline.models_circuits_and_piplines.models.baseline_models.predict_hybrid_model import PHModel

full_config = {
    'time_frame_start': -4 * np.pi,
    'time_frame_end': 8 * np.pi,
    'n_steps': 256,
    'time_steps': 64,
    'future_steps': 6,
    'num_samples': 256,
    'noise_level': 0.05,
    'train_test_ratio': 0.6,
    'preview_samples': 3,
    'show_dataset_plots': False,
    'show_model_plots': False,
    'show_forecast_plots': True,
    'model': 'Amp_circuit',
    'circuit': 'layered_Amp_Circuit',
    'epochs': 1,
    'batch_size': 32,
    'learning_rate': 0.08,
    'loss_function': 'mse',
    'steps_to_predict': 300,
    'compress_factor': 1.5,
    'layers': 24,
    'wires': 10
}

models = {'Hybrid': PHModel, 'Variable_circuit': PVCModel, 'Amp_circuit': PACModel}
circuits = {'new_RYXZ_Circuit': new_RYXZ_Circuit, 'new_baseline': new_baseline, 'base_Amp_Circuit': base_Amp_Circuit,
            'layered_Amp_Circuit': layered_Amp_Circuit, 'Test_Circuit': Test_Circuit}

def function(x):
    return np.sin(x) + 0.5 * np.cos(2 * x) + 0.25 * np.sin(3 * x)

def run_model(dataset, config):
    circuit = circuits[config['circuit']]
    model = models[config['model']](circuit, config)

    model_save_path = os.path.join(filemanager.path, "fitted_model")

    logger.info("Starting training")
    loss_progress = model.train(dataset)
    logger.info("Training completed")

    logger.info("Starting evaluation")
    pred_y_test_data, loss = model.evaluate(dataset)
    logger.info(f"Test Loss: {loss}")

    show_all_evaluation_plots(pred_y_test_data, loss_progress, dataset, config)

    return model, loss

def main():
    logger.info("Generating training data")
    dataset = generate_time_series_data(function, full_config)
    logger.info("Training data generated")

    model = run_model(dataset, full_config)

    iterative_forecast(function, model, dataset, full_config)

if __name__ == "__main__":
    start_time = time.time()
    silence_tensorflow()
    main()
    logger.info(f"Pipeline complete in {time.time() - start_time} seconds")
    logger.info(f"Config of this run: {full_config}")
