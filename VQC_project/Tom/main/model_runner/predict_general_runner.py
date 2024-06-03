# Needs to be first import
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from VQC_project.Tom.main.div.n_training_data_manager import generate_time_series_data
from VQC_project.Tom.main.model.predict_hybrid.predict_hybrid_metrics import plot_metrics, plot_predictions, \
    plot_residuals
from VQC_project.Tom.main.model.predict_classic.n_predict_model import PCModel
from VQC_project.Tom.main.model.predict_quantum.predict_quantum_model import PQModel
from VQC_project.Tom.main.model.predict_hybrid.predict_hybrid_model import PHModel

config = {
    # training data parameter
    'time_frame_start': -8 * np.pi,  # start of timeframe
    'time_frame_end': 8 * np.pi,  # end of timeframe, needs to be bigger than time_frame_start
    'data_length': 100,  # How many points are in the full timeframe
    'time_steps': 5,  # How many consecutive points are in train/test sample
    'future_steps': 5,  # How many points are predicted in train/test sample
    'num_samples': 300,  # How many samples of time_steps/future_steps are generated from the timeframe
    'noise_level': 0,  # Noise level on Inputs
    'train_test_ratio': 0.5,  # The higher the ratio to more data is used for training
    # run parameter
    'epochs': 5,  # Adjusted to start with a reasonable number
    'batch_size': 8,  # Keep this value for now
    'input_dim': 1,  # Currently stays at one
    # Q_layer parameter
    'n_qubits': 5,  # Amount of wires used, when using the Quantum Model: n_qubits = time_steps
    'n_layers': 5,  # Amount of strongly entangled layers in
    # Optimization parameter
    'learning_rate': 0.004,  # Adjusted to a common starting point
    'loss_function': 'mse',  # currently at 'mse'
}

active_model = PHModel(config)  # Hybrid_Model


# active_model = PCModel(config) # Classic_Model
# active_model = PQModel(config) # Quantum_Model


def function(x):
    return np.sin(x) + 0.5 * np.cos(2 * x) + 0.25 * np.sin(3 * x)


def main(target_function, model):
    # Generate training data
    dataset = generate_time_series_data(config, target_function)

    # Samples that are used as input, continuous set of y_values
    input_train = dataset['Input_train']
    # Samples that should be predicted, continuous set of y_values that follow after the input
    output_train = dataset['Output_train']
    # Train the model
    history = model.train(input_train, output_train)

    # Samples that are used as input, not seeing before by the model
    input_test = dataset['Input_test']
    # Samples that should be predicted, continuous set of y_values that follow after the input, not seeing before by
    # the model
    output_test = dataset['Output_test']
    # Evaluate how well it learned the test_data
    loss = model.evaluate(input_test, output_test)
    print(f"Test Loss: {loss}")

    # Predict outputs for test inputs
    pred_y_test_data = model.predict(input_test)
    print(f"Test Predictions: {pred_y_test_data}")

    # Last known real y_values, used to make predict unknown y values
    input_future = dataset['Input_future']
    # Predict outputs for last known y values

    pred_y_data_future_data = model.predict(input_future)
    print(f"Test Predictions: {pred_y_data_future_data}")
    # Generate x-axes based on indices for consistency
    output_test_indices = np.arange(len(output_test))
    output_future_indices = np.arange(config['future_steps'])

    # Alternatively, use time-based range for both
    output_test_time_based = np.linspace(config['time_frame_start'], config['time_frame_end'],
                                         len(output_test.flatten()))
    x_future_time = np.linspace(config['time_frame_start'], config['time_frame_end'], config['future_steps'])

    # Plot training metrics
    plot_metrics(history)

    # Plot predictions vs real values for test set
    for i in range(len(pred_y_test_data)):
        output_test_indices = np.arange(output_test[i])
        plot_predictions(output_test_indices, output_test[i], pred_y_test_data[i],
                     title='Test Data: Real vs Predicted')

    # Plot predictions vs real values for future data
    plot_predictions(output_future_indices, function(output_future_indices), pred_y_data_future_data.flatten(),
                     title='Future Data: Real vs Predicted')

    # Alternatively, if using time-based range
    plot_predictions(output_test_time_based, output_test.flatten(), pred_y_test_data.flatten(),
                     title='Test Data: Real vs Predicted (Time-Based)')

    plot_predictions(x_future_time, function(x_future_time), pred_y_data_future_data.flatten(),
                     title='Future Data: Real vs Predicted (Time-Based)')

    # Plot residuals for test data
    plot_residuals(output_test.flatten(), pred_y_test_data.flatten(), title='Residuals on Test Data')


if __name__ == "__main__":
    from silence_tensorflow import silence_tensorflow

    silence_tensorflow()
    main(function, active_model)
