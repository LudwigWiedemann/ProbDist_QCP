# Needs to be first import
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from time import sleep
import numpy as np

from VQC_project.Tom.main.div.n_training_data_manager import generate_time_series_data
from VQC_project.Tom.main.model.predict_hybrid.predict_hybrid_metrics import plot_metrics, plot_predictions, plot_residuals
from VQC_project.Tom.main.model.predict_classic.n_predict_model import PCModel
from VQC_project.Tom.main.model.predict_quantum.predict_quantum_model import PQModel
from VQC_project.Tom.main.model.predict_hybrid.predict_hybrid_model import PHModel

config = {
    # training data parameter
    'time_frame_start': -4 * np.pi,  # start of timeframe
    'time_frame_end': 4 * np.pi,  # end of timeframe, needs to be bigger than time_frame_start
    'data_length': 150,  # How many points are in the full timeframe
    'time_steps': 5,  # How many consecutive points are in train/test sample
    'future_steps': 5,  # How many points are predicted in train/test sample
    'num_samples': 1000,  # How many samples of time_steps/future_steps are generated from the timeframe
    'noise_level': 0.1,  # Noise level on Inputs
    'train_test_ratio': 0.6,  # The higher the ratio to more data is used for training
    # run parameter
    'model': 'Hybrid',
    'epochs': 30,  # Adjusted to start with a reasonable number
    'batch_size': 64,  # Keep this value for now
    'input_dim': 1,  # Currently stays at one
    # Q_layer parameter
    'n_qubits': 8,  # Amount of wires used, when using the Quantum Model: n_qubits = time_steps
    'n_layers': 8,  # Amount of strongly entangled layers in
    # Optimization parameter
    'learning_rate': 0.003,  # Adjusted to a common starting point
    'loss_function': 'mse',  # currently at 'mse'
    # Forcasting parameter
    'steps_to_predict': 25
}

models = {'Hybrid': PHModel(config), 'Classic': PCModel(config), 'Quantum': PQModel(config)}

def function(x):
    return np.sin(x) + 0.5 * np.cos(2 * x) + 0.25 * np.sin(3 * x)


def iterative_forecast(model, initial_input, steps, future_steps, time_steps):
    current_input = initial_input
    all_predictions = []

    for _ in range(steps // future_steps):
        # Ensure the current input has the correct shape
        if current_input.shape[1] < time_steps:
            padding = np.zeros((current_input.shape[0], time_steps - current_input.shape[1], current_input.shape[2]))
            current_input = np.concatenate((padding, current_input), axis=1)

        pred = model.predict(current_input)
        all_predictions.append(pred.flatten())
        # Use the last `time_steps` of the combined current_input + prediction as the next input
        current_input = np.concatenate((current_input.flatten(), pred.flatten()))[-time_steps:].reshape(1, -1, 1)

    return np.concatenate(all_predictions)


def main(target_function):
    print("Generating training data")
    # Generate training data
    dataset = generate_time_series_data(config, target_function)
    print("Training data generated")
    model = models[config['model']]
    # Samples that are used as input, continuous set of y_values
    input_train = dataset['Input_train']
    # Samples that should be predicted, continuous set of y_values that follow after the input
    output_train = dataset['Output_train']
    # Train the model
    print("Starting training")
    history = model.train(input_train, output_train)
    print("Training completed")

    # Samples that are used as input, not seen before by the model
    input_test = dataset['Input_test']
    # Samples that should be predicted, continuous set of y_values that follow after the input, not seen before by
    # the model
    output_test = dataset['Output_test']
    # Evaluate how well it learned the test_data
    print("Starting evaluation")
    loss = model.evaluate(input_test, output_test)
    print(f"Test Loss: {loss}")

    # Predict outputs for test inputs
    print("Predicting test data")
    pred_y_test_data = model.predict(input_test)

    # Last known real y_values, used to make predict unknown y values
    input_future = dataset['Input_future']
    # Predict outputs for last known y values
    print("Predicting future data")
    pred_y_data_future_data = model.predict(input_future)

    # Generate x-axes based on indices for consistency
    output_future_indices = np.arange(config['time_steps'] + config['future_steps'])

    # Plot training metrics
    plot_metrics(history)

    # Plot predictions vs real values for each test sample
    for i in range(len(pred_y_test_data)):
        if i % config['num_samples'] / 10 == 0:
            x_indices = np.arange(config['time_steps'] + config['future_steps'])
            y_real_combined = np.concatenate((input_test[i].flatten(), output_test[i].flatten()))
            y_pred_combined = np.concatenate((input_test[i].flatten(), pred_y_test_data[i].flatten()))
            plot_predictions(x_indices, input_test[i].flatten(), y_real_combined, y_pred_combined,
                             noise_level=config['noise_level'],
                             title=f'Test Data Sample {i + 1}: Real vs Predicted')
            sleep(1.5)

    # Plot predictions vs real values for future data
    plot_predictions(output_future_indices, input_future.flatten(),
                     np.concatenate((input_future.flatten(), pred_y_data_future_data.flatten())),
                     np.concatenate((input_future.flatten(), pred_y_data_future_data.flatten())),
                     noise_level=config['noise_level'],
                     title='Future Data: Real vs Predicted')

    # Plot residuals for test data
    plot_residuals(output_test.flatten(), pred_y_test_data.flatten(), title='Residuals on Test Data')

    # Iterative forecasting
    iter_predictions = iterative_forecast(model, input_future, config['steps_to_predict'], config['future_steps'],
                                          config['time_steps'])

    # Calculate all real future values at once
    real_future_values = function(np.linspace(config['time_frame_end'], config['time_frame_end'] + config['steps_to_predict'], config['steps_to_predict']))

    # Generate x-axes for iterative predictions
    x_iter_indices = np.arange(config['time_steps'] + config['steps_to_predict'])
    y_iter_combined = np.concatenate((input_future.flatten(), real_future_values))
    plot_predictions(x_iter_indices, input_future.flatten(), y_iter_combined,
                     np.concatenate((input_future.flatten(), iter_predictions)), noise_level=config['noise_level'],
                     title='Iterative Forecast: Real vs Predicted', marker_distance=10)


if __name__ == "__main__":
    from silence_tensorflow import silence_tensorflow

    silence_tensorflow()
    main(function)
