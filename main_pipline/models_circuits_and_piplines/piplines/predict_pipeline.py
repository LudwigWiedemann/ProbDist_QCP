import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import random
from main_pipline.models_circuits_and_piplines.piplines.predict_pipline_div.predict_iterative_forecasting import \
    iterative_forecast
from main_pipline.models_circuits_and_piplines.piplines.predict_pipline_div.predict_shot_forecaste import \
    evaluate_sample_with_shot, iterative_shot_forecast



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

trial_name = 'classic_output'

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
    'show_dataset_plots': False,
    'show_model_plots': False,
    'show_forecast_plots': True,
    'show_approx_plots': True,
    'steps_to_predict': 300,
    # Model parameter
    'model': 'PSCModel',
    'circuit': 'Tangle_Shot_Circuit',
    # Run parameter
    'epochs': 1,
    'batch_size': 37,
    'learning_rate': 0.0094,
    'loss_function': 'mse',
    'compress_factor':  4.116,
    'patience': 40,
    'min_delta': 0.001,
    # Circuit parameter
    'layers': 2,  # Only Optuna/Tangle circuit
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
    'Tangle_Shot_Circuit':Tangle_Shot_Circuit
}


def function(x):
    return np.sin(x) + 0.5 * np.cos(2 * x) + 0.25 * np.sin(3 * x)


def run_model(dataset, config, logger):
    circuit = circuits[config['circuit']]
    model = models[config['model']](circuit, config)
    logger.info(f"Model: {config['model']}, Circuit: {config['circuit']}")
    logger.info("Config of this run:")
    for key in config.keys():
        logger.info(f"{key} : {config[key]}")
    model_save_path = os.path.join(logger.folder_path, "fitted_model")

    logger.info("Starting training")
    loss_progress = model.train(dataset, logger)
    logger.info("Training completed")

    logger.info("Starting evaluation")
    pred_y_test_data, loss = model.evaluate(dataset)
    logger.info(f"Test Loss: {loss}")

    show_all_evaluation_plots(pred_y_test_data, loss_progress, dataset, config, logger)
    #model.save_model(model_save_path, logger)
    return model, loss


def main():
    trial_folder = filemanager.create_folder(trial_name)
    logger = Logger(trial_folder)

    logger.info("Generating training data")
    dataset = generate_dataset(function, full_config, logger)
    logger.info("Training data generated")

    model, loss = run_model(dataset, full_config, logger)

    #iterative_forecast(function, model, dataset, full_config, logger=logger)

    logger.info("Start Shot_sample_forecasting")
    n_shots = [5, 1000, 10000]#, 100000]#, 1000000]
    sample_index = random.sample(range(len(dataset['input_test'])), full_config['approx_samples'])
    for shots in n_shots:
        logger.info(f'Evaluating Sample with {shots} shots')
        evaluate_sample_with_shot(model, dataset, sample_index, full_config, logger, title=shots, custome_shots=shots)
    for shots in n_shots:
        logger.info(f'Evaluating Forecast with {shots} shots')
        shots_start = time.time()
        iterative_shot_forecast(function, model, dataset, full_config, logger=logger, title=shots, custome_shots=shots)
        logger.info(f"Shot_Forecast with {shots} took {time.time() - shots_start}")


    logger.info(f"Pipeline complete in {time.time() - start_time} seconds")


if __name__ == "__main__":
    start_time = time.time()
    silence_tensorflow()
    main()
