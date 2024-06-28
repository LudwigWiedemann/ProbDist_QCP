import os
import time
import dill
import multiprocessing
import numpy as np
import optuna
from silence_tensorflow import silence_tensorflow
from main_pipline.input.div.dataset_manager import generate_time_series_data
from main_pipline.models_circuits_and_piplines.piplines.predict_pipeline import run_model
from main_pipline.models_circuits_and_piplines.piplines.predict_pipline_div.predict_iterative_forecasting import iterative_forecast

n_trials = 100

# Get the current directory of this script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the location for saving the study progress
saved_progress_file = os.path.join(CURRENT_DIR, "optuna_study_01.pkl")
db_url = f"sqlite:///{os.path.join(CURRENT_DIR, 'optuna_study.db')}"

# Save the study every n trials
SAVE_INTERVAL = 1

full_config = {
    # Dataset parameter
    'time_frame_start': -2 * np.pi,
    'time_frame_end': 5 * np.pi,
    'n_steps': 256,
    'time_steps': 64,
    'future_steps': 6,
    'num_samples': 256,
    'noise_level': 0.05,
    'train_test_ratio': 0.6,
    #  Plotting parameter
    'preview_samples': 3,
    'show_dataset_plots': False,
    'show_model_plots': False,
    'show_forecast_plots': True,
    'steps_to_predict': 300,
    # Model parameter
    'model': 'Amp_circuit',
    'circuit': 'Tangle_Amp_Circuit',
    # Run parameter
    'epochs': 40,
    'batch_size': 32,
    'learning_rate': 0.5,
    'loss_function': 'mse',
    'compress_factor': 1.5,
    'patience': 10,
    'min_delta': 0.001,
    # Circuit parameter
    'layers': 5,  # Only Optuna/Tangle circuit
    'shots': None
}

def function(x):
    return np.sin(x) + 0.5 * np.cos(2 * x) + 0.25 * np.sin(3 * x)

full_dataset = generate_time_series_data(function, full_config)

def optimize(trial):
    silence_tensorflow()
    layers = trial.suggest_int('layers', 10, 25)
    learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.1)
    compress_factor = trial.suggest_float('compress_factor', 1, 10)
    batch_size = trial.suggest_int('batch_size', 1, 128)

    print(f"Trial {trial.number}: Start optimization with parameters: layers={layers}, learning_rate={learning_rate}, compress_factor={compress_factor}")
    dataset = full_dataset
    config = full_config
    config.update([('layers', layers), ('learning_rate', learning_rate), ('compress_factor', compress_factor), ('batch_size', batch_size)])
    model, loss = run_model(dataset, config)
    iterative_forecast(function, model, dataset, config)
    print(f"Trial {trial.number}: Finished with parameters: layers={layers}, learning_rate={learning_rate}, compress_factor={compress_factor}, batch_size={batch_size}")
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

    pruner = optuna.pruners.MedianPruner()
    sampler = optuna.samplers.RandomSampler()

    storage = optuna.storages.RDBStorage(url=db_url)
    study = optuna.create_study(storage=storage, sampler=sampler, pruner=pruner, direction="maximize")

    # Load logs if they exist
    print(f"Trying to load study file from: {saved_progress_file}", flush=True)
    if os.path.exists(saved_progress_file):
        print(f"Study file found at: {saved_progress_file}", flush=True)
        with open(saved_progress_file, 'rb') as f:
            study = dill.load(f)
        print("Study loaded.", flush=True)
    else:
        print(f"No previous study found at {saved_progress_file}. Starting a new one.", flush=True)

    # Define a callback function to save the study after each trial
    def save_study_callback(study, trial):
        if trial.number % SAVE_INTERVAL == 0:
            save_study(study, saved_progress_file)

    study.optimize(optimize, n_trials=n_trials, callbacks=[save_study_callback])

    print(f"Time needed for config optimisation {time.time() - start_time}", flush=True)
    print(f"Best trial: {study.best_trial}", flush=True)
    print(f"Best parameters: {study.best_trial.params}", flush=True)

    # Save the final study
    save_study(study, saved_progress_file)

def run_parallel_optimisation():
    # Determine the number of available CPU cores
    num_workers = multiprocessing.cpu_count()

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
