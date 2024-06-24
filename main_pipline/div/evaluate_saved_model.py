import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from main_pipline.input.div.logger import logger as log
from main_pipline.input.div.dataset_manager import generate_time_series_data
from main_pipline.models_circuits_and_piplines.piplines.predict_pipline_div.predict_plots_and_metrics import \
    show_all_evaluation_plots

# Configuration dictionary
eval_config = {
    'model_save_path': 'fitted_model',
    'steps_to_predict': 300
}


def function(x):
    return np.sin(x) + 0.5 * np.cos(2 * x) + 0.25 * np.sin(3 * x)


def load_model(path):
    return tf.keras.models.load_model(path, custom_objects={'KerasLayer': qml.qnn.KerasLayer})


def evaluate_model(model, dataset, config):
    x_test = dataset['input_test']
    y_test = dataset['output_test']

    # Predict on the test set
    pred_y_test_data = model.predict(x_test)

    # Calculate loss
    loss = model.evaluate(x_test, y_test)
    log.info(f"Test Loss: {loss}")

    # Plotting predictions vs actual values
    plt.figure(figsize=(14, 7))
    plt.plot(y_test.flatten(), label='Actual')
    plt.plot(pred_y_test_data.flatten(), label='Predicted')
    plt.title('Actual vs Predicted')
    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.legend()
    plt.show()

    # Print plots related to the evaluation of test data
    show_all_evaluation_plots(pred_y_test_data, [], dataset, config)


def main():
    log.info("Generating evaluation data")
    dataset = generate_time_series_data(function, eval_config)
    log.info("Evaluation data generated")

    log.info("Loading model")
    model = load_model(eval_config['model_save_path'])
    log.info("Model loaded")

    log.info("Starting evaluation")
    evaluate_model(model, dataset, eval_config)
    log.info("Evaluation completed")


if __name__ == "__main__":
    main()
