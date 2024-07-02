import os
import json
import numpy as np
import tensorflow as tf
import pennylane as qml

from main_pipline.models_circuits_and_piplines.circuits.amp_Circuit import (
    base_Amp_Circuit, layered_Amp_Circuit, tangle_Amp_Circuit,
    test_Amp_Circuit, double_Amp_Circuit
)
from main_pipline.models_circuits_and_piplines.circuits.variable_circuit import (
    new_RYXZ_Circuit, new_baseline
)
from main_pipline.models_circuits_and_piplines.models.baseline_models.predict_classic_circuit_model import PCCModel
from main_pipline.models_circuits_and_piplines.models.baseline_models.predict_hybrid_model import PHModel
from main_pipline.models_circuits_and_piplines.models.predict_amp_circuit_model import PACModel
from main_pipline.models_circuits_and_piplines.models.predict_variable_circuit_model import PVCModel
from main_pipline.models_circuits_and_piplines.piplines.predict_pipline_div.predict_iterative_forecasting import (
    iterative_forecast
)
from main_pipline.input.div.logger import Logger
from main_pipline.input.div.dataset_manager import generate_time_series_data
from main_pipline.models_circuits_and_piplines.piplines.predict_pipline_div.predict_plots_and_metrics import (
    show_all_evaluation_plots
)

models = {
    'PHModel': PHModel,
    'PVCModel': PVCModel,
    'PACModel': PACModel,
    'PCCModel': PCCModel
}

circuits = {
    'new_RYXZ_Circuit': new_RYXZ_Circuit,
    'new_baseline': new_baseline,
    'base_Amp_Circuit': base_Amp_Circuit,
    'layered_Amp_Circuit': layered_Amp_Circuit,
    'Tangle_Amp_Circuit': tangle_Amp_Circuit,
    'Test_Circuit': test_Amp_Circuit,
    'Double_Amp_Circuit': double_Amp_Circuit
}


def function(x):
    return np.sin(x) + 0.5 * np.cos(2 * x) + 0.25 * np.sin(3 * x)


def evaluate_model(model_path, config_path):
    trial_name = 'Model_Evaluation'
    trial_folder = os.path.join(os.getcwd(), trial_name)
    os.makedirs(trial_folder, exist_ok=True)
    logger = Logger(trial_folder)

    # Load the model
    logger.info("Loading model")
    try:
        model = tf.keras.models.load_model(model_path, custom_objects={'KerasLayer': qml.qnn.KerasLayer})
    except Exception as e:
        logger.warning(f"Failed to load model with error: {e}. Trying to load weights instead.")
        model, config = load_model_weights(model_path, config_path)
    else:
        with open(config_path, 'r') as f:
            config = json.load(f)

    logger.info("Model loaded")

    # Generate dataset
    logger.info("Generating evaluation data")
    dataset = generate_time_series_data(function, config, logger)
    logger.info("Evaluation data generated")

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


def load_model_weights(model_path, config_path):
    with open(config_path, 'r') as f:
        config = json.load(f)

    circuit = circuits[config['circuit']]
    model_class = models[config['model']]
    model = model_class(circuit, config)
    model.model.load_weights(model_path)
    return model, config


if __name__ == "__main__":
    model_path = os.path.join("..", "output", "Quantum_Model_test", "fitted_model")
    config_path = os.path.join("..", "output", "Quantum_Model_test", "fitted_model_config.json")
    evaluate_model(model_path, config_path)
