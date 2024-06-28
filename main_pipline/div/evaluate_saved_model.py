import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from main_pipline.input.div.logger import logger as log
from main_pipline.input.div.dataset_manager import generate_time_series_data
from main_pipline.models_circuits_and_piplines.piplines.predict_pipline_div.predict_plots_and_metrics import show_all_evaluation_plots
from main_pipline.models_circuits_and_piplines.models.predict_amp_circuit_model import PACModel
from main_pipline.models_circuits_and_piplines.circuits.amp_Circuit import tangle_Amp_Circuit

# Configuration dictionary
eval_config = {
    'model_save_path': 'main_pipline/output/PD output from 28.06.2024--13-36-07',  # Path without the extension
    'steps_to_predict': 300,
    # Other configuration parameters need to match those used during training
    'time_frame_start': -4 * np.pi,
    'time_frame_end': 12 * np.pi,
    'n_steps': 80,
    'time_steps': 8,
    'future_steps': 4,
    'num_samples': 80,
    'noise_level': 0.1,
    'train_test_ratio': 0.6,
    #  Plotting parameter
    'preview_samples': 3,
    'show_dataset_plots': True,
    'show_model_plots': False,
    'show_forecast_plots': True,
    'steps_to_predict': 300,
    # Model parameter
    'model': 'Amp_circuit',
    'circuit': 'Tangle_Amp_Circuit',
    # Run parameter
    'epochs': 1,
    'batch_size': 32,
    'learning_rate': 0.5,
    'loss_function': 'mse',
    'compress_factor': 1.5,
    'patience': 10,
    'min_delta': 0.001,
    # Circuit parameter
    'layers': 2,  # Only Optuna/Tangle circuit
    'shots': None
}

def function(x):
    return np.sin(x) + 0.5 * np.cos(2 * x) + 0.25 * np.sin(3 * x)

def load_model(path, config):
    if not os.path.exists(path + ".index") or not os.path.exists(path + ".data-00000-of-00001"):
        log.error(f"Model files not found in the path: {path}")
        raise FileNotFoundError(f"Model files not found in the path: {path}")

    circuit = tangle_Amp_Circuit(config)
    model = PACModel(circuit, config)
    model.load_model(path)
    return model

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
    model = load_model(eval_config['model_save_path'], eval_config)
    log.info("Model loaded")

    log.info("Starting evaluation")
    evaluate_model(model, dataset, eval_config)
    log.info("Evaluation completed")

if __name__ == "__main__":
    main()
