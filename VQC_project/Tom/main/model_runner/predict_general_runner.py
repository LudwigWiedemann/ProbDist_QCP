# Needs to be first import to
import os

from VQC_project.Tom.main.div.n_training_data_manager import generate_time_series_data
from VQC_project.Tom.main.model_runner.predict_general_runner_div.iterative_forecasting import iterative_forecast

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np


# Different Models
from VQC_project.Tom.main.model.predict_classic.predict_model import PCModel
from VQC_project.Tom.main.model.predict_quantum.predict_quantum_model import PQModel
from VQC_project.Tom.main.model.predict_hybrid.predict_hybrid_model import PHModel
from VQC_project.Tom.main.model.predict_variable_circuit.predict_variable_circuit_model import PVCModel

full_config = {
    # training data parameter
    'time_frame_start': -4 * np.pi,  # start of timeframe
    'time_frame_end': 4 * np.pi,  # end of timeframe, needs to be bigger than time_frame_start
    'data_length': 150,  # How many points are in the full timeframe
    'time_steps': 60,  # How many consecutive points are in train/test sample
    'future_steps': 10,  # How many points are predicted in train/test sample
    'num_samples': 1000,  # How many samples of time_steps/future_steps are generated from the timeframe
    'noise_level': 0.1,  # Noise level on Inputs
    'train_test_ratio': 0.6,  # The higher the ratio to more data is used for training
    # Run parameter
    'model': 'Hybrid',
    'circuit':'RYXZ_ciuit',
    'epochs': 50,  # Adjusted to start with a reasonable number
    'batch_size': 64,  # Keep this value for now
    # Optimization parameter
    'learning_rate': 0.01,  # Adjusted to a common starting point
    'loss_function': 'mse',  # currently at 'mse'
    # Forcasting parameter
    'steps_to_predict': 25
}

models = {'Hybrid': PHModel, 'Classic': PCModel, 'Quantum': PQModel, 'Variable_circuit': PVCModel}

def function(x):
    return np.sin(x) + 0.5 * np.cos(2 * x) + 0.25 * np.sin(3 * x)


def run_model(dataset, config, circuit=None):
    # Initialised the current model
    model = models[config['model']](config)

    # Samples used for training
    input_train = dataset['Input_train']
    output_train = dataset['Output_train']
    # Samples used for testing
    input_test = dataset['Input_test']
    output_test = dataset['Output_test']


    # Fit the model
    print("Starting training")
    history = model.train(input_train, output_train)
    print("Training completed")

    # Evaluate how well it learned the test_data
    print("Starting evaluation")
    loss = model.evaluate(input_test, output_test)
    print(f"Test Loss: {loss}")

    # Predict outputs for test inputs
    print("Predicting test data")
    pred_y_test_data = model.predict(input_test)

    # Print plots related to the evaluation of test data
    print_test_plots(input_test, output_test, pred_y_test_data, history, config)

    return model


def main():
    # Generate dataset
    print("Generating training data")
    dataset = generate_time_series_data(full_config, function)
    print("Training data generated")
    # Train and evaluate model and plot results
    model = run_model(dataset, full_config)
    # Forecast using the fitted model and plot results
    iterative_forecast(model, dataset, full_config)


if __name__ == "__main__":
   # from silence_tensorflow import silence_tensorflow

   # silence_tensorflow()

    main()
