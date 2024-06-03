# Needs to be first import
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from VQC_project.Tom.main.div.n_training_data_manager import generate_time_series_data
from VQC_project.Tom.main.model.predict_hybrid.predict_hybrid_metrics import plot_metrics, plot_predictions, plot_residuals, compare_multiple_predictions
from VQC_project.Tom.main.model.predict_hybrid.predict_hybrid_model import PHModel

config = {
    # training data parameter
    'time_frame_start': -8 * np.pi,
    'time_frame_end': 8 * np.pi,
    'data_length': 300,  # How many points are in the full timeframe
    'time_steps': 200,  # How many consecutive points are in one sample
    'future_steps': 10,  # How many points are predicted in one sample
    'num_samples': 100,  # How many timeframes are generated from the timeframe
    'noise_level': 0,  # Noise level
    'train_test_ratio': 0.6,  # The higher the ratio to more data is used for training
    # run parameter
    'epochs': 100,  # Adjusted to start with a reasonable number
    'batch_size': 16,  # Keep this value for now
    'input_dim': 1,
    # Q_layer parameter
    'n_qubits': 5,
    'n_layers': 5,
    # Optimization parameter
    'learning_rate': 0.004,  # Adjusted to a common starting point
    'loss_function': 'mse',
}
def target_function(x):
    return np.sin(x) + 0.5 * np.cos(2 * x) + 0.25 * np.sin(3 * x)

def main():
    # Generate training data
    x_train, y_train, x_test, y_test, x_future = generate_time_series_data(config, np.sin)

    # Initialize the model
    model = PHModel(config)

    # Train the model
    history = model.train(x_train, y_train)

    # Evaluate the model
    loss = model.evaluate(x_test, y_test)
    print(f"Test Loss: {loss}")

    # Make predictions on the test set
    y_pred_test = model.predict(x_test)

    # Make predictions on the future data
    y_pred_future = model.predict(x_future)

    # Generate x-axes based on indices for consistency
    x_test_indices = np.arange(len(y_test.flatten()))
    x_future_indices = np.arange(config['future_steps'])

    # Alternatively, use time-based range for both
    x_test_time = np.linspace(config['time_frame_start'], config['time_frame_end'], len(y_test.flatten()))
    x_future_time = np.linspace(config['time_frame_start'], config['time_frame_end'], config['future_steps'])

    # Plot training metrics
    plot_metrics(history)

    # Plot predictions vs real values for test set
    plot_predictions(x_test_indices, y_test.flatten(), y_pred_test.flatten(),
                     title='Test Data: Real vs Predicted')

    # Plot predictions vs real values for future data
    plot_predictions(x_future_indices, np.sin(x_future_indices), y_pred_future.flatten(),
                     title='Future Data: Real vs Predicted')

    # Alternatively, if using time-based range
    plot_predictions(x_test_time, y_test.flatten(), y_pred_test.flatten(),
                     title='Test Data: Real vs Predicted (Time-Based)')

    plot_predictions(x_future_time, np.sin(x_future_time), y_pred_future.flatten(),
                     title='Future Data: Real vs Predicted (Time-Based)')

    # Plot residuals for test data
    plot_residuals(y_test.flatten(), y_pred_test.flatten(), title='Residuals on Test Data')


if __name__ == "__main__":
    from silence_tensorflow import silence_tensorflow

    silence_tensorflow()
    main()
