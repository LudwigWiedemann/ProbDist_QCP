from pennylane import numpy as np
import tensorflow as tf

def generate_training_data(target_function, config):
    # Choose random points in range
    x_train = np.linspace(config['range_start'], config['range_end'], config['num_points'])

    # Generate y values and add noise
    y_train = target_function(x_train)
    noise = config['noise_level'] * np.random.normal(size=x_train.shape)
    y_train = y_train + noise

    # Reshape x into vector
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
    x_train = tf.reshape(x_train, (-1, 1))
    y_train = tf.reshape(y_train, (-1, 1))
    return x_train, y_train
