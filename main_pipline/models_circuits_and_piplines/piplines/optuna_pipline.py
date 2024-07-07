import os
import time
import dill
import multiprocessing
import numpy as np
import optuna
from silence_tensorflow import silence_tensorflow
from main_pipline.input.div.dataset_manager import generate_time_series_data
from main_pipline.models_circuits_and_piplines.piplines.predict_pipeline import run_model

import main_pipline.input.div.filemanager as filemanager
from main_pipline.input.div.logger import Logger
from main_pipline.models_circuits_and_piplines.piplines.predict_pipline_div.predict_iterative_forecasting import \
    iterative_forecast
from main_pipline.models_circuits_and_piplines.piplines.predict_pipline_div.predict_shot_forecaste import \
    iterative_shot_forecast

trial_name = 'Major_Test_v3'
n_trials = 200
num_workers = 5  # Set the number of workers here

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

saved_progress_file = os.path.join(CURRENT_DIR, f"{trial_name}.pkl")
db_url = f"sqlite:///{os.path.join(CURRENT_DIR, f'{trial_name}.db')}"

SAVE_INTERVAL = 1

full_config = {
    # Dataset parameter
    'time_frame_start': -4 * np.pi,
    'time_frame_end': 12 * np.pi,
    'n_steps': 256,
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
    'epochs': 50,
    'batch_size': 55,
    'learning_rate': 0.03,
    'loss_function': 'mse',
    'compress_factor': 8.61,
    'patience': 40,
    'min_delta': 0.001,
    # Circuit parameter
    'layers': 22,  # Only Optuna/Tangle circuit
    # Shot prediction
    'approx_samples': 2,
    'shots': 10000,
    'shot_predictions': 100,
}


def function(x):
    return np.sin(x) + 0.5 * np.cos(2 * x) + 0.25 * np.sin(3 * x)


def generate_dataset(logger):
    try:
        dataset = generate_time_series_data(function, full_config, logger)
        logger.info("Training data generated")
        return dataset
    except Exception as e:
        logger.error(f"Error generating dataset: {e}")
        raise


def optimize(trial):
    silence_tensorflow()
    layers = trial.suggest_int('layers', 10, 25)
    learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.1)
    compress_factor = trial.suggest_float('compress_factor', 1, 10)
    batch_size = trial.suggest_int('batch_size', 1, 64)
    n_steps = trial.suggest_int('n_steps', 70, 256)

    trial_folder = filemanager.create_folder(
        f"{trial_name}_{trial.number}_la{layers}_lr{np.round(learning_rate, 4)}_cf{np.round(compress_factor, 2)}_bs{batch_size}")
    logger = Logger(trial_folder)

    logger.info(
        f"Trial {trial.number}: Start optimization with parameters: layers={layers}, learning_rate={learning_rate}, compress_factor={compress_factor}")

    dataset = generate_dataset(logger)

    config = full_config.copy()
    config.update({
        'layers': layers,
        'learning_rate': learning_rate,
        'compress_factor': compress_factor,
        'batch_size': batch_size,
        'n_steps': n_steps
    })

    model, loss = run_model(dataset, config, logger)

    iterative_forecast(function, model, dataset, config, logger=logger)
    iterative_shot_forecast(function, model, dataset, full_config, logger=logger)

    logger.info(
        f"Trial {trial.number}: Finished with parameters: layers={layers}, learning_rate={learning_rate},"
        f" compress_factor={compress_factor}, batch_size={batch_size}")
    return evaluate_dataset_results(loss)


def evaluate_dataset_results(loss):
    return loss


def save_study(study, file_path):
    with open(file_path, 'wb') as f:
        dill.dump(study, f)
    print(f"Study saved to {file_path}", flush=True)


def run_optimisation():
    start_time = time.time()
    print("Start config optimisation", flush=True)

    # pruner = optuna.pruners.MedianPruner()
    sampler = optuna.samplers.TPESampler()

    storage = optuna.storages.RDBStorage(url=db_url)

    print(f"Trying to load study file from: {saved_progress_file}", flush=True)
    try:
        study = optuna.load_study(study_name=trial_name, storage=storage)
        print("Study loaded.", flush=True)
    except KeyError:
        print(f"No previous study found under the name {trial_name}. Starting a new one.", flush=True)
        study = optuna.create_study(study_name=trial_name, storage=storage, sampler=sampler,
                                    direction="minimize")

    def save_study_callback(study, trial):
        if trial.number % SAVE_INTERVAL == 0:
            save_study(study, saved_progress_file)

    study.optimize(optimize, n_trials=n_trials, callbacks=[save_study_callback])

    print(f"Time needed for config optimisation {time.time() - start_time}", flush=True)
    print(f"Best trial: {study.best_trial}", flush=True)
    print(f"Best parameters: {study.best_trial.params}", flush=True)

    save_study(study, saved_progress_file)


def run_parallel_optimisation():
    processes = []
    for _ in range(num_workers):
        p = multiprocessing.Process(target=run_optimisation)
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


def main():
    silence_tensorflow()
    run_parallel_optimisation()


if __name__ == "__main__":
    main()
