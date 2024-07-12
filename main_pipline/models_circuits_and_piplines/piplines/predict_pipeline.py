import os
import random

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from main_pipline.models_circuits_and_piplines.piplines.predict_pipline_div.predict_iterative_forecasting import \
    fully_iterative_forecast, partial_iterative_forecast
from main_pipline.models_circuits_and_piplines.piplines.predict_pipline_div.predict_shot_forecaste import \
    fully_iterative_shot_forecast, partial_iterative_shot_forecast, evaluate_sample_with_shot
from main_pipline.models_circuits_and_piplines.circuits.shots_Circuit import test_Shot_Circuit, Tangle_Shot_Circuit
from main_pipline.models_circuits_and_piplines.models.predict_shots_circuit_model import PSCModel

import time
from silence_tensorflow import silence_tensorflow
import numpy as np
import main_pipline.input.div.filemanager as filemanager
from main_pipline.input.div.logger import Logger

from main_pipline.input.div.dataset_manager import generate_dataset
from main_pipline.models_circuits_and_piplines.circuits.variable_circuit import new_RYXZ_Circuit, new_baseline
from main_pipline.models_circuits_and_piplines.circuits.amp_Circuit import base_Amp_Circuit, layered_Amp_Circuit, \
    tangle_Amp_Circuit, test_Amp_Circuit, double_Amp_Circuit
from main_pipline.models_circuits_and_piplines.models.baseline_models.predict_classic_model import PCModel
from main_pipline.models_circuits_and_piplines.models.predict_variable_circuit_model import PVCModel
from main_pipline.models_circuits_and_piplines.models.predict_amp_circuit_model import PACModel
from main_pipline.models_circuits_and_piplines.models.baseline_models.predict_hybrid_model import PHModel
from main_pipline.models_circuits_and_piplines.piplines.predict_pipline_div.predict_plots_and_metrics import \
    show_all_evaluation_plots

trial_name = 'Quantum_computing_old_settings'

full_config = {
    # Dataset parameter
    'time_frame_start': -4 * np.pi,
    'time_frame_end': 12 * np.pi,
    'n_steps': 171,
    'time_steps': 64,
    'future_steps': 6,
    'num_samples': 256,
    'noise_level': 0.05,
    'train_test_ratio': 0.6,
    # Plotting parameter
    'preview_samples': 3,
    'show_dataset_plots': True,
    'show_model_plots': True,
    'show_forecast_plots': True,
    'show_approx_plots': True,
    'steps_to_predict': 300,
    # Model parameter
    'model': 'PSCModel',
    'circuit': 'Tangle_Shot_Circuit',
    # Run parameter
    'epochs': 50,
    'batch_size': 37,
    'learning_rate': 0.079,
    'loss_function': 'mse',
    'compress_factor': 4.116,
    'patience': 40,
    'min_delta': 0.001,
    # Circuit parameter
    'layers': 16,  # Only Optuna/Tangle circuit
    # Shot prediction
    'approx_samples': 2,
    'shots': 100000,
    'shot_predictions': 100,
}

models = {
    'PHModel': PHModel,
    'PVCModel': PVCModel,
    'PACModel': PACModel,
    'PCModel': PCModel,
    'PSCModel': PSCModel
}

circuits = {
    'new_RYXZ_Circuit': new_RYXZ_Circuit,
    'new_baseline': new_baseline,
    'base_Amp_Circuit': base_Amp_Circuit,
    'layered_Amp_Circuit': layered_Amp_Circuit,
    'Tangle_Amp_Circuit': tangle_Amp_Circuit,
    'Test_Circuit': test_Amp_Circuit,
    'Double_Amp_Circuit': double_Amp_Circuit,
    'test_Shot_Circuit': test_Shot_Circuit,
    'Tangle_Shot_Circuit': Tangle_Shot_Circuit
}


def function(x):
    return np.sin(x) + 0.5 * np.cos(2 * x) + 0.25 * np.sin(3 * x)


def create_unique_folder_name(base_name):
    counter = 0
    unique_name = base_name
    while os.path.exists(unique_name):
        counter += 1
        unique_name = f"{base_name}_{counter}"
    return unique_name


def run_model(dataset, config, logger):
    circuit = circuits[config['circuit']]
    model = models[config['model']](circuit, config)
    model.print_circuit(config['circuit'])

    logger.info(f"Model: {config['model']}, Circuit: {config['circuit']}")
    logger.info("Config of this run:")
    for key in config.keys():
        logger.info(f"{key} : {config[key]}")

    logger.info("Starting training")
    loss_progress = model.train(dataset, logger)
    logger.info("Training completed")

    logger.info("Starting evaluation")
    pred_y_test_data, loss = model.evaluate(dataset)
    logger.info(f"Test Loss: {loss}")

    show_all_evaluation_plots(pred_y_test_data, loss_progress, dataset, config, logger)

    # model_save_path = os.path.join(logger.folder_path, "fitted_model")
    # model.save_model(model_save_path, logger)
    return model, loss


def main():
    trial_folder = create_unique_folder_name(trial_name)
    trial_folder = filemanager.create_folder(trial_folder)
    logger = Logger(trial_folder)

    logger.info("Generating training data")
    dataset = generate_dataset(function, full_config, logger)
    logger.info("Training data generated")

    model, loss = run_model(dataset, full_config, logger)

    fully_iterative_forecast(model, dataset, full_config, logger=logger)
    partial_iterative_forecast(model, dataset, full_config, logger=logger)

    logger.info("Start Shot_sample_forecasting")
    n_shots = [5, 1000, 10000, 100000, 1000000]
    n_predictions = [1, 5, 10, 50, 100]
    sample_index = random.sample(range(len(dataset['input_test'])), full_config['approx_samples'])
    for prediction in n_predictions:
        full_config.update({'shot_predictions': prediction})
        for shots in n_shots:
            logger.info(f'Evaluating Sample with {shots} shots with {prediction} prediction')
            evaluate_sample_with_shot(model, dataset, sample_index, full_config, logger,
                                      title=f'{shots}_shots_{prediction}_predictions', custome_shots=shots)
        for shots in n_shots:
            logger.info(f'Evaluating Forecasting with {shots} shots with {prediction} prediction')
            shots_start = time.time()
            fully_iterative_shot_forecast(model, dataset, full_config, logger=logger,
                                          title=f'{shots}_shots_{prediction}_predictions', custome_shots=shots)
            partial_iterative_shot_forecast(model, dataset, full_config, logger=logger,
                                            title=f'{shots}_shots_{prediction}_predictions', custome_shots=shots)
            logger.info(f"Shot_Forecast with {shots} took {time.time() - shots_start}")

    logger.info(f"Pipeline complete in {time.time() - start_time} seconds")


if __name__ == "__main__":
    start_time = time.time()
    silence_tensorflow()
    main()
