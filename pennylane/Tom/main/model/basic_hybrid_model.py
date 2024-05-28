import tensorflow as tf
from pennylane import numpy as np

from model.div.metric_tools import plot_results
from model.div.model_shape_tools import create_quantum_circuit, create_model


def train_hybrid_model(training_data, config):
    quantum_circuit, weight_shapes = create_quantum_circuit(config)

    model = create_model(quantum_circuit, weight_shapes, config)
    x_train, y_train = training_data
    model.fit(x_train, y_train, epochs=200, batch_size=20, verbose=1)
    return model


def evaluate_model(target_function, model, training_data):
    # Predict further values
    x_test = np.linspace(-3 * np.pi, 6 * np.pi, 100)
    y_test = target_function(x_test)
    x_test = tf.reshape(tf.convert_to_tensor(x_test), (-1, 1))
    y_pred = model.predict(x_test)

    # Plot results
    x_train, y_train = training_data
    plot_results(x_test, y_test, x_train, y_train, y_pred)
