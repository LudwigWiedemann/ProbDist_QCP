import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import random

from main_pipline.models_circuits_and_piplines.piplines.predict_pipline_div.predict_iterative_forecasting import \
    fully_iterative_forecast, partial_iterative_forecast
from main_pipline.models_circuits_and_piplines.piplines.predict_pipline_div.predict_shot_forecaste import \
    fully_iterative_shot_forecast, partial_iterative_shot_forecast, evaluate_sample_with_shot
from main_pipline.models_circuits_and_piplines.circuits.shots_Circuit import test_Shot_Circuit, Tangle_Shot_Circuit, \
    Reup_Shot_Circuit, Ludwig2_Shot_Circuit
from main_pipline.models_circuits_and_piplines.models.predict_shots_circuit_model import PSCModel
from main_pipline.models_circuits_and_piplines.piplines.predict_pipline_div.distribution_calculator import calculate_distribution_with_KLD
import time
from silence_tensorflow import silence_tensorflow
from pennylane import numpy as np
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

trial_name = 'Ludwig_computing'

full_config = {
    # Dataset parameter
    'time_frame_start': 0,
    'time_frame_end': 10,
    'n_steps': 174,
    'time_steps': 64,
    'future_steps': 6,
    'num_samples': 10,
    'noise_level': 0.05,
    'train_test_ratio': 0.6,
    # Plotting parameter
    'preview_samples': 3,
    'show_dataset_plots': True,
    'show_model_plots': True,
    'show_forecast_plots': True,
    'show_approx_plots': True,
    'steps_to_predict': 30,
    # Model parameter
    'model': 'PSCModel',
    'circuit': 'Ludwig2_Shot_Circuit',
    # Run parameter
    'epochs': 5,
    'batch_size': 16,
    'learning_rate': 0.03,
    'loss_function': 'mse',
    'compress_factor': 1,
    'patience': 10,
    'min_delta': 0.001,
    # Circuit parameter
    'layers': 1,  # Only Optuna/Tangle circuit
    # Shot prediction
    'approx_samples': 3,
    'shots': 10,
    'shot_predictions': 300,
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
    'Tangle_Shot_Circuit': Tangle_Shot_Circuit,
    'Reup_Shot_Circuit': Reup_Shot_Circuit,
    'Ludwig2_Shot_Circuit': Ludwig2_Shot_Circuit
}


def function(x):
    return np.sin(x) + 0.5 * np.cos(2 * x) + 0.25 * np.sin(3 * x)


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

    # Placeholder for computing the loss on evaluation data
    # You need to replace this with actual logic to compute loss
    loss = "computed_loss_placeholder"

    # Placeholder for generating predictions on test data
    # You need to replace this with actual logic to generate predictions
    pred_y_test_data = "pred_y_test_data_placeholder"

    # Uncomment and adjust the following lines according to your actual evaluation logic
    # logger.info("Starting evaluation")
    # pred_y_test_data, loss = model.evaluate(dataset)
    # logger.info(f"Test Loss: {loss}")
    # show_all_evaluation_plots(pred_y_test_data, loss_progress, dataset, config, logger)

    return model, loss, pred_y_test_data


def main():
    trial_folder = filemanager.create_folder(trial_name)
    logger = Logger(trial_folder)

    logger.info("Generating training data")
    dataset = generate_dataset(function, full_config, logger)
    logger.info("Training data generated")

    model, loss, pred_y_test_data = run_model(dataset, full_config, logger)

    fully_kl_output = fully_iterative_shot_forecast(model, dataset, full_config)
    step_size = 1

    partial_kl_output = partial_iterative_shot_forecast(model, dataset, full_config)


    # Ensure dataset is an iterable (e.g., list of datasets)
    if not isinstance(dataset, (list, tuple)):
        dataset = [dataset]  # Wrap dataset in a list if it's not already an iterable

    calculate_distribution_with_KLD(fully_kl_output, dataset[0]['extended_y_data'], step_size, full_config['time_frame_start'], full_config['time_frame_end']+full_config['steps_to_predict'],logger)


    logger.info(f"Pipeline complete in {time.time() - start_time} seconds")

if __name__ == "__main__":
    start_time = time.time()
    silence_tensorflow()
    main()
