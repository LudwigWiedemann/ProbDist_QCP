import os
import random

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
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
    show_all_evaluation_plots, show_all_shot_forecasting_plots

trial_name = 'Ludwig_computing'

full_config = {
    # Dataset parameter
    'time_frame_start': 0,
    'time_frame_end': 10,
    'n_steps': 20,
    'time_steps': 8,
    'future_steps': 2,
    'num_samples': 100,
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
    'circuit': 'Ludwig2_Shot_Circuit',
    # Run parameter
    'epochs': 80,
    'batch_size': 16,
    'learning_rate': 0.03,
    'loss_function': 'mse',
    'compress_factor': 1,
    'patience': 10,
    'min_delta': 0.001,
    # Circuit parameter
    'layers': 5,  # Only Optuna/Tangle circuit
    # Shot prediction
    'approx_samples': 3,
    'shots': 100,
    'shot_predictions': 20,
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

    logger.info("Starting evaluation")
    pred_y_test_data, loss = model.evaluate(dataset)
    logger.info(f"Test Loss: {loss}")

    show_all_evaluation_plots(pred_y_test_data, loss_progress, dataset, config, logger)

    # model_save_path = os.path.join(logger.folder_path, "fitted_model")
    # model.save_model(model_save_path, logger)
    return model, loss, pred_y_test_data


def main():
    """
    Main function to run the pipeline
    :return: None
    """
    # Create a new folder for the trial and a new logger
    trial_folder = filemanager.create_folder(trial_name)
    logger = Logger(trial_folder)

    logger.info("Generating training data")
    dataset = generate_dataset(function, full_config, logger)
    logger.info("Training data generated")

    model, loss, pred_y_test_data = run_model(dataset, full_config, logger)

    fully_iterative_forecast(model, dataset, full_config, logger=logger)
    partial_iterative_forecast(model, dataset, full_config, logger=logger)

    logger.info("Start Shot_sample_forecasting")
    # Evaluate the model with different number of shots and predictions
    n_shots = [10000]
    n_predictions = [50]
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

            fully_outputs = fully_iterative_shot_forecast(model, dataset, full_config, custome_shots=shots)

            show_all_shot_forecasting_plots(fully_outputs, dataset, full_config, logger=logger,
                                            title=f'Fully_Iterative_Forecast_{shots}_shots_{prediction}_predictions')
            step_size=1
            partial_outputs = partial_iterative_shot_forecast(model, dataset, full_config, custome_shots=shots)

            show_all_shot_forecasting_plots(partial_outputs, dataset, full_config, logger=logger,
                                            title=f'Partial_Iterative_Forecast_{shots}_shots_{prediction}_predictions')
            # Ensure dataset is an iterable (e.g., list of datasets)
            if not isinstance(dataset, (list, tuple)):
                datasets = [dataset]  # Wrap dataset in a list if it's not already an iterable
            else: datasets = dataset
            # Calculate the KL divergence between the fully and partially iterative forecasts
            calculate_distribution_with_KLD(fully_outputs, datasets[0]['extended_y_data'], step_size, full_config['time_frame_start'], full_config['time_frame_end'],logger)
    logger.info(f"Shot_Forecast with {shots} took {time.time() - shots_start}")
    logger.info(f"Pipeline complete in {time.time() - start_time} seconds")


if __name__ == "__main__":
    start_time = time.time()
    silence_tensorflow()
    main()
