import tensorflow as tf
from pennylane import numpy as np

from VQC_project.Tom.main.model.predict_classic.predict_classic_metric import plot_pc_results
from VQC_project.Tom.main.model.predict_classic.predict_classic_mode_shape import create_pc_model


def train_pc_model(training_data, config,x,y):

    model = create_pc_model(config)

    x_train, y_train = training_data
    model.fit(x_train, y_train, epochs=config['epochs'], batch_size=config['batch_size'], verbose=1)
    return model


def evaluate_pc_model(target_function, model, training_data):
    x_test = np.linspace(-3 * np.pi, 6 * np.pi, 100)
    y_test = target_function(x_test)
    x_test = tf.reshape(tf.convert_to_tensor(x_test), (-1, 1))
    y_pred = model.predict(x_test)

    x_train, y_train = training_data

    plot_pc_results(x_test, y_test, x_train, y_train, y_pred)
