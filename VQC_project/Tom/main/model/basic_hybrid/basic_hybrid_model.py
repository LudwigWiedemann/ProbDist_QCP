
import datetime
import tensorflow as tf
from keras.src.callbacks import TensorBoard
from pennylane import numpy as np

from VQC_project.Tom.main.model.basic_hybrid.basic_hybrid_metric import Bh_TrainingPlot, plot_bh_results
from VQC_project.Tom.main.model.basic_hybrid.basic_hybrid_model_shape import create_bh_quantum_circuit, create_bh_model


def train_hybrid_model(training_data, config):
    # TensorBoard callback
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)
    # Custom TrainingPlot callback
    training_plot_callback = Bh_TrainingPlot()

    quantum_circuit, weight_shapes = create_bh_quantum_circuit(config)
    model = create_bh_model(quantum_circuit, weight_shapes, config)

    x_train, y_train = training_data
    model.fit(x_train, y_train, epochs=config['epochs'], batch_size=config['batch_size'], verbose=1,
              callbacks=[tensorboard_callback, training_plot_callback])
    return model


def evaluate_model(target_function, model, training_data):
    # Predict further values
    x_test = np.linspace(-3 * np.pi, 6 * np.pi, 100)
    y_test = target_function(x_test)
    x_test = tf.reshape(tf.convert_to_tensor(x_test), (-1, 1))
    y_pred = model.predict(x_test)

    # Plot results
    x_train, y_train = training_data
    plot_bh_results(x_test, y_test, x_train, y_train, y_pred)
