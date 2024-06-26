import os
import sys
import time
import dill
import numpy as np
import optuna
from silence_tensorflow import silence_tensorflow

from main_pipline.input.div.dataset_manager import generate_time_series_data
from main_pipline.models_circuits_and_piplines.piplines.predict_pipeline import run_model

n_trials = 100

# Get the current directory of this script
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

# Define the location for saving the study progress
saved_progress_file = os.path.join(CURRENT_DIR, "optuna_study_01.pkl")

# Save the study every n trials
SAVE_INTERVAL = 5

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
    'epochs': 100,
    'batch_size': 32,
    'learning_rate': 0.08,
    'loss_function': 'mse',
    'steps_to_predict': 300,
    'compress_factor': 1.5,
    'patience': 10,
    'min_delta': 0.001
}

def function(x):
    return np.sin(x) + 0.5 * np.cos(2 * x) + 0.25 * np.sin(3 * x)

full_dataset = generate_time_series_data(function, full_config)

def optimize(trial):
    layers = trial.suggest_int('layers', 10, 50)
    learning_rate = trial.suggest_float('learning_rate', 0.0001, 0.1)
    compress_factor = trial.suggest_float('compress_factor', 1, 10)
    batch_size = trial.suggest_int('batch_size', 1, 128)

    print(f"Trial {trial.number}: Start optimization with parameters: layers={layers}, learning_rate={learning_rate}, compress_factor={compress_factor}")
    dataset = full_dataset
    config = full_config
    config.update([('layers', layers), ('learning_rate', learning_rate), ('compress_factor', compress_factor), ('batch_size', batch_size)])
    model, loss = run_model(dataset, config)

    print(f"Trial {trial.number}: Finished with parameters: layers={layers}, learning_rate={learning_rate}, compress_factor={compress_factor}, batch_size={batch_size}")
    return evaluate_dataset_results(loss)

def evaluate_dataset_results(loss):
    return loss

def run_optimisation():
    start_time = time.time()
    print("Start config optimisation", flush=True)

    pruner = optuna.pruners.MedianPruner()
    sampler = optuna.samplers.RandomSampler()

    study = optuna.create_study(sampler=sampler, pruner=pruner, direction="maximize")
    # Load logs if they exist
    print(f"Trying to load study file from: {saved_progress_file}", flush=True)
    if os.path.exists(saved_progress_file):
        print(f"Study file found at: {saved_progress_file}", flush=True)
        with open(saved_progress_file, 'rb') as f:
            study = dill.load(f)
        print("Study loaded.", flush=True)
    else:
        print(f"No previous study found at {saved_progress_file}. Starting a new one.", flush=True)

    study.optimize(optimize, n_trials=n_trials)  # Use all available CPU cores

    print(f"Time needed for config optimisation {time.time() - start_time}", flush=True)
    print(f"Best trial: {study.best_trial}", flush=True)
    print(f"Best parameters: {study.best_trial.params}", flush=True)

    # Save the final study
    with open(saved_progress_file, 'wb') as f:
        dill.dump(study, f)
    print(f"Final study saved to {saved_progress_file}", flush=True)

def main():
    silence_tensorflow()
    run_optimisation()

if __name__ == "__main__":
    main()
