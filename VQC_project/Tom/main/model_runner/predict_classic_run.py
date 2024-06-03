# Needs to be first import
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import numpy as np
from VQC_project.Tom.main.div.n_training_data_manager import generate_time_series_data
from VQC_project.Tom.main.model.predict_classic.n_predict_classic_metrics import plot_metrics, plot_predictions
from VQC_project.Tom.main.model.predict_classic.n_predict_model import PCModel

config = {
    # training data parameter
    'data_length': 600,
    'time_steps': 120,  # Ensure this is less than the length of x_data in generate_time_series_data
    'num_samples': 100,
    'noise_level': 0.2,  # Adjust noise level as needed
    'future_steps': 80,  # Number of future steps for prediction
    # run parameter
    'epochs': 200,  # Adjusted to start with a reasonable number
    'batch_size': 16,  # Keep this value for now
    'input_dim': 1,
    # Optimization parameter
    'learning_rate': 0.005,  # Adjusted to a common starting point
    'loss_function': 'mse',
}

def main():
    # Generate training data
    x_train, y_train, x_test, y_test, x_future = generate_time_series_data(config, np.sin)

    # Initialize the model
    model = PCModel(config)

    # Train the model
    history = model.train(x_train, y_train)

    # Evaluate the model
    loss = model.evaluate(x_test, y_test)
    print(f"Test Loss: {loss}")

    # Make predictions on the test set
    y_pred_test = model.predict(x_test)

    # Make predictions on the future data
    y_pred_future = model.predict(x_future)
    x_future_flat = np.linspace(0, 8 * np.pi, config['future_steps'])  # Match future range length

    # Debug: Print shapes and values
    print("Shapes and some values for debugging:")
    print(f"x_future_flat.shape: {x_future_flat.shape}")
    print(f"x_future_flat: {x_future_flat}")
    print(f"y_pred_future.flatten().shape: {y_pred_future.flatten().shape}")
    print(f"y_pred_future.flatten(): {y_pred_future.flatten()}")
    print(f"np.sin(x_future_flat).shape: {np.sin(x_future_flat).shape}")
    print(f"np.sin(x_future_flat): {np.sin(x_future_flat)}")

    # Plot training metrics
    plot_metrics(history)

    # Plot predictions vs real values for test set
    plot_predictions(np.arange(len(y_test.flatten())), y_test.flatten(), y_pred_test.flatten(),
                     title='Test Data: Real vs Predicted')

    # Plot predictions vs real values for future data
    plot_predictions(x_future_flat, np.sin(x_future_flat), y_pred_future.flatten(),
                     title='Future Data: Real vs Predicted')


if __name__ == "__main__":
    from silence_tensorflow import silence_tensorflow

    silence_tensorflow()
    main()
