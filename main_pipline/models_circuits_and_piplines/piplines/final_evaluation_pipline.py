import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import time
from pennylane import numpy as np
import random
import tensorflow as tf

from main_pipline.models_circuits_and_piplines.piplines.predict_pipeline import run_model
from main_pipline.models_circuits_and_piplines.piplines.predict_pipline_div.predict_iterative_forecasting import \
    fully_iterative_forecast, partial_iterative_forecast
from main_pipline.models_circuits_and_piplines.piplines.predict_pipline_div.predict_shot_forecaste import \
    fully_iterative_shot_forecast, partial_iterative_shot_forecast, evaluate_sample_with_shot

from silence_tensorflow import silence_tensorflow
import main_pipline.input.div.filemanager as filemanager
from main_pipline.input.div.logger import Logger
from main_pipline.input.div.dataset_manager import generate_dataset, function1, function2, function3, function4, \
    function5, function0
from main_pipline.models_circuits_and_piplines.piplines.predict_pipline_div.predict_plots_and_metrics import (
    show_all_seed_evaluation_plots)

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
    'show_model_plots': False,
    'show_forecast_plots': True,
    'show_approx_plots': False,
    'steps_to_predict': 300,
    # Model parameter
    'model': 'PSCModel',
    'circuit': 'Ludwig2_Shot_Circuit',
    # Run parameter
    'epochs': 150,
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

function_dictionary = {
    "function_0": function0,
    "function_1": function1,
    "function_2": function2,
    "function_3": function3,
    "function_4": function4,
    "function_5": function5
}


def classic_setup(config):
    config.update({'time_frame_start': -12.566370614359172,
                   'time_frame_end': 37.69911184307752,
                   'n_steps': 171,
                   'time_steps': 64,
                   'future_steps': 6,
                   'num_samples': 256,
                   # Model parameter
                   'model': 'PCModel',
                   # Run parameter
                   'epochs': 400,
                   'batch_size': 37,
                   'learning_rate': 0.005,
                   'compress_factor': 4.116,
                   'layers': 16
                   })
    return config, False


def quantum_baseline_setup(config):
    config.update({'time_frame_start': -12.566370614359172,
                   'time_frame_end': 37.69911184307752,
                   'n_steps': 171,
                   'time_steps': 64,
                   'future_steps': 6,
                   'num_samples': 256,
                   # Model parameter
                   'model': 'PSCModel',
                   'circuit': 'Tangle_Shot_Circuit',
                   # Run parameter
                   'batch_size': 37,
                   'learning_rate': 0.079,
                   'compress_factor': 4.116,
                   'layers': 16
                   })
    return config, True


def quantum_team_setup_compression(config):
    config.update({'time_frame_start': 0,
                   'time_frame_end': 10,
                   'n_steps': 20,
                   'time_steps': 8,
                   'future_steps': 2,
                   'num_samples': 100,
                   # Model parameter
                   'model': 'PSCModel',
                   'circuit': 'Ludwig2_Shot_Circuit',
                   # Run parameter
                   'batch_size': 15,
                   'learning_rate': 0.04,
                   'compress_factor': 8.52880998496839,
                   'layers': 1
                   })
    return config, True


def quantum_team_setup_legacy(config):
    config.update({'time_frame_start': 0,
                   'time_frame_end': 10,
                   'n_steps': 20,
                   'time_steps': 8,
                   'future_steps': 2,
                   'num_samples': 100,
                   # Model parameter
                   'model': 'PSCModel',
                   'circuit': 'Ludwig2_Shot_Circuit',
                   # Run parameter
                   'batch_size': 16,
                   'learning_rate': 0.03,
                   'compress_factor': 1,
                   'layers': 5
                   })
    return config, True


setup_dictionary = {
    #"Classic": classic_setup,
    #"Baseline": quantum_baseline_setup,
    "Team": quantum_team_setup_compression,
    "Team_legacy": quantum_team_setup_legacy
}


def setseeds(seed):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def close_logger(logger):
    logger.close()


def evaluate_circuit_performance(trial_name, dataset, config, hasShots=False):
    silence_tensorflow()
    trial_folder = filemanager.create_folder(trial_name)
    logger = Logger(trial_folder)

    model, loss, pred_y_test_data = run_model(dataset, config, logger)

    fully_output_ar = fully_iterative_forecast(model, dataset, config, logger=logger)
    partial_output_ar = partial_iterative_forecast(model, dataset, config, logger=logger)

    if hasShots:
        logger.info("Start Shot_sample_forecasting")
        n_shots = [5, 100, 1000, 10000, 100000, 1000000]
        n_predictions = [1, 10, 100]
        sample_index = random.sample(range(len(dataset['input_test'])), config['approx_samples'])
        for prediction in n_predictions:
            full_config.update({'shot_predictions': prediction})

            for shots in n_shots:
                logger.info(f'Evaluating Sample with {shots} shots with {prediction} prediction')
                evaluate_sample_with_shot(model, dataset, sample_index, config, logger,
                                          title=f'{shots}_shots_{prediction}_predictions', custome_shots=shots)

            for shots in n_shots:
                logger.info(f'Evaluating Forecasting with {shots} shots with {prediction} prediction')
                shots_start = time.time()
                fully_iterative_shot_forecast(model, dataset, full_config, logger=logger,
                                              title=f'{shots}_shots_{prediction}_predictions', custome_shots=shots)
                partial_iterative_shot_forecast(model, dataset, full_config, logger=logger,
                                                title=f'{shots}_shots_{prediction}_predictions', custome_shots=shots)
                logger.info(f"Shot_Forecast with {shots} took {time.time() - shots_start}")
    results = {'loss': loss, 'fully_output_ar': fully_output_ar, 'partial_output_ar': partial_output_ar}
    close_logger(logger)
    return results


def main():
    for setup_key in setup_dictionary.keys():
        config = full_config.copy()
        config, hasShots = setup_dictionary[setup_key](config)
        for function_key in function_dictionary.keys():
            trial_folder = filemanager.create_folder(f"Final_Evaluation_{setup_key}_{function_key}")
            seed_logger = Logger(trial_folder)

            loss_outputs = []
            fully_outputs = []
            partial_outputs = []

            dataset = generate_dataset(function_dictionary[function_key], config, seed_logger)
            for i in range(5):
                setseeds(i)
                seed_logger.info(f'Model: {config["model"]}, Circuit: {config["circuit"]}')
                seed_logger.info(f'Config of this run: {config}')
                seed_logger.info(f'Starting training')
                results = evaluate_circuit_performance(trial_name=f'{setup_key}_{function_key}_{i}', dataset=dataset,
                                                       config=config, hasShots=hasShots)
                seed_logger.info(f'Run {i + 1} of 5:')
                seed_logger.info(f"Loss: {results['loss']}")
                loss_outputs.append(results['loss'])
                fully_outputs.append(results['fully_output_ar'])
                partial_outputs.append(results['partial_output_ar'])
            close_logger(seed_logger)
            show_all_seed_evaluation_plots(fully_outputs, dataset, config, seed_logger, title='Fully_')
            show_all_seed_evaluation_plots(partial_outputs, dataset, config, seed_logger, title='Partial')


if __name__ == "__main__":
    main()
