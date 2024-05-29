#Needs to be before
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


import sys
import time
from silence_tensorflow import silence_tensorflow
from pennylane import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
main_dir = os.path.abspath(os.path.join(current_dir, '..'))
os.chdir(main_dir)
sys.path.insert(0, main_dir)

from div.training_data_manager import generate_training_data
from model.basic_hybrid_model import train_hybrid_model, evaluate_model

config = {
    # training data parameter
    'num_points': 200,
    'range_start': 10,
    'range_end': 0,
    'noise_level': 0.3,
    # run parmeter
    'epochs': 100,
    'batch_size': 15,
    # Q_layer parameter
    'n_qubits': 5,
    'n_layers': 5,
    # Optimisation parameter
    'learning_rate': 0.05,
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
