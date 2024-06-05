# Needs to be first import to
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
import pennylane as qml

from VQC_project.Tom.main.div.training_data_manager import generate_time_series_data
from VQC_project.Tom.main.div.plot_manager import print_test_plots, print_iterative_forcast_plot
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


def create_custom_circuit():
    dev = qml.device("default.qubit", wires=1, interface='tf')

    @qml.qnode(dev)
    def quantum_circuit(inputs, weights):
        qml.RX(inputs[0], wires=0)
        qml.RX(weights[1], wires=0)
        qml.RY(weights[2], wires=0)
        qml.RZ(weights[3], wires=0)
        return qml.expval(qml.PauliZ(0))


    input_shape = []
    weight_shapes = []
    return quantum_circuit, weight_shapes


def generate_dataset(target_function, config):
    # Generate training data
    print("Generating training data")
    dataset = generate_time_series_data(config, target_function)
    print("Training data generated")
    return dataset


def run_model(dataset, config, circuit=None):
    # Initialised the current model
    model = models[config['model']](config)

    # Samples used for training
    input_train = dataset['Input_train']
    output_train = dataset['Output_train']
    # Samples used for testing
    input_test = dataset['Input_test']
    output_test = dataset['Output_test']
    # Samples used for forecasting
    input_future = dataset['Input_future']

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


def iterative_forecast(model, dataset, config):
    steps = config['steps_to_predict']
    time_steps = config['time_steps']
    future_steps = config['future_steps']

    initial_input = dataset['Future_Input']
    # Start of prediction
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
    print_iterative_forcast_plot(function, initial_input, np.concatenate(all_predictions), config)


def main():
    # Generate dataset
    dataset = generate_dataset(function, full_config)
    # Train and evaluate model and plot results
    model = run_model(dataset, full_config, circuit=None)
    # Forecast using the fitted model and plot results
    iterative_forecast(model, dataset, full_config)


if __name__ == "__main__":
    from silence_tensorflow import silence_tensorflow

    silence_tensorflow()

    main()
