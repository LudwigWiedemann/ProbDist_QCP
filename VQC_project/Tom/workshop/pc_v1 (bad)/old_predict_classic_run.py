# Needs to be first import
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import time
from silence_tensorflow import silence_tensorflow
from pennylane import numpy as np
from VQC_project.Tom.main.div.training_data_manager import create_time_based_dataset
from VQC_project.Tom.main.model.predict_classic.predict_classic_model import train_pc_model, evaluate_pc_model
import tensorflow as tf
import numpy as np


def create_time_based_dataset(target_function, config):
    # Generate the x values and corresponding y values with noise
    x = np.linspace(0, 100, config['n_points'])
    y = target_function(x) + config['noise_level'] * np.random.normal(size=config['n_points'])

    # Create the dataset using efficient array slicing
    dataX = np.array([y[i:(i + config['time_steps'])] for i in range(len(x) - config['time_steps'])])
    dataY = y[config['time_steps']:]

    # Convert the dataset to TensorFlow tensors
    return tf.convert_to_tensor(dataX, dtype=tf.float32), tf.convert_to_tensor(dataY, dtype=tf.float32)

config = {
    # training data parameter
    'time_steps': 100,
    'n_points': 1000,
    'noise_level': 0.3,
    # run parameter
    'epochs': 10,
    'batch_size': 15,
    'input_dim': 1,
    # Optimization parameter
    'learning_rate': 0.001,
    'loss_function': 'mse',
}

def target_function(x):
    return np.sin(x)
    #np.sin(x) + 0.5 * np.cos(2 * x) + 0.25 * np.sin(3 * x)


def main():
    start_time = time.time()
    # Generate training data
    training_data = create_time_based_dataset(target_function, config)
    x_train, y_train = training_data
    print(f"x_train shape: {x_train.shape}, y_train shape: {y_train.shape}")

    # Generate test data
    x_test = np.linspace(-3 * np.pi, 6 * np.pi, 100)
    y_test = target_function(x_test)
    x_test = tf.reshape(tf.convert_to_tensor(x_test), (-1, 1))

    # Train the model
    model = train_pc_model(training_data, config, x_test, y_test)

    # Evaluate the model
    evaluate_pc_model(target_function, model, training_data)

    print(f"Total computation time: {time.time() - start_time}")


if __name__ == "__main__":
    silence_tensorflow()
    main()
