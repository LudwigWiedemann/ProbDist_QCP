# Needs to be before
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from VQC_project.Tom.main.div.training_data_manager import generate_training_data
from VQC_project.Tom.main.model.basic_hybrid.basic_hybrid_model import train_hybrid_model, evaluate_model



import time
from silence_tensorflow import silence_tensorflow
from pennylane import numpy as np

config = {
    # training data parameter
    'n_points': 200,
    'range_start': 10,
    'range_end': 0,
    'noise_level': 0.3,
    # run parameter
    'epochs': 100,
    'batch_size': 15,
    # Q_layer parameter
    'n_qubits': 5,
    'n_layers': 5,
    # Optimization parameter
    'learning_rate': 0.001,
    'loss_function': 'mse',
}

def target_function(x):
    return np.sin(x) + 0.5 * np.cos(2 * x) + 0.25 * np.sin(3 * x)

def main():
    start_time = time.time()
    # Generate training data
    training_data = generate_training_data(target_function, config)
    # Train the model
    model = train_hybrid_model(training_data, config)
    # Evaluate the model
    evaluate_model(target_function, model, training_data)

    print(f"Total computation time: {time.time() - start_time}")

if __name__ == "__main__":
    # Mutes all tf warnings, specifically the losing the complex part of the VQC layer one
    silence_tensorflow()
    main()
