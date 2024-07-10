import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from main_pipline.models_circuits_and_piplines.piplines.predict_pipline_div.predict_iterative_forecasting import \
    iterative_forecast
import numpy as np
import tensorflow as tf
from main_pipline.input.div.logger import Logger
from main_pipline.input.div.dataset_manager import generate_time_series_data
from main_pipline.models_circuits_and_piplines.piplines.predict_pipline_div.predict_plots_and_metrics import \
    show_all_evaluation_plots
import pennylane as qml


# Configuration similar to the original
config = {
    'time_frame_start': -4 * np.pi,
    'time_frame_end': 12 * np.pi,
    'n_steps': 256,
    'time_steps': 64,
    'future_steps': 6,
    'num_samples': 256,
    'noise_level': 0.5,
    'train_test_ratio': 0.6,
    'preview_samples': 3,
    'show_dataset_plots': False,
    'show_model_plots': False,
    'show_forecast_plots': True,
    'steps_to_predict': 300,
    'batch_size': 55,
    'compress_factor': 8.61
}


def function(x):
    return np.sin(x) + 0.5 * np.cos(2 * x) + 0.25 * np.sin(3 * x)


def evaluate_model(model_path, config):
    trial_name = 'Model_Evaluation'
    trial_folder = os.path.join(os.getcwd(), trial_name)
    os.makedirs(trial_folder, exist_ok=True)
    logger = Logger(trial_folder)

    # Generate dataset
    logger.info("Generating evaluation data")
    dataset = generate_time_series_data(function, config, logger)
    logger.info("Evaluation data generated")

    # Load the model
    logger.info("Loading model")
    model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': qml.qnn.KerasLayer})
    logger.info("Model loaded")

    # Extract the relevant parts of the dataset for evaluation
    x_test = dataset['input_test']
    y_test = dataset['output_test']

    # Evaluate the model
    logger.info("Starting evaluation")
    loss = model.evaluate(x_test, y_test)
    logger.info(f"Test Loss: {loss}")

    # Predict and show evaluation plots
    predictions = model.predict(x_test)
    show_all_evaluation_plots(predictions, {'loss': [loss]}, dataset, config, logger=logger)
    iterative_forecast(function, model, dataset, config, logger=logger)


if __name__ == "__main__":
    # Use absolute path
    model_path = os.path.abspath(os.path.join("..", "output", "Quantum_Model_test", "fitted_model"))
    evaluate_model(model_path, config)
