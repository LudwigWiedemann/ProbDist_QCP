from pennylane import numpy as np
import training as tr
import plotting as plot
import noise as ns


full_config = {
    # data parameter
    'x_start': 0,
    'x_end': 100,
    'total_training_points': 110,
    # training parameter
    'time_steps': 10,  # How many consecutive points are in train/test sample
    'future_steps': 10,  # How many points are predicted in train/test sample
    'num_weights': 10,
    'num_samples': 100,  # How many samples of time_steps/future_steps are generated from the timeframe
    'noise_level': 0.1,  # Noise level on Inputs
    'train_test_ratio': 0.6,  # The higher the ratio to more data is used for training
    'epochs': 20,  # Adjusted to start with a reasonable number
    'learning_rate': 0.001,  # Adjusted to a common starting point
    # Forecasting parameter
    'steps_to_forecast': 50
}



def prepare_data():
    training_time_steps = np.linspace(full_config['x_start'], full_config['x_end'],
                                      full_config['total_training_points'])
    training_dataset = [tr.f(x) for x in training_time_steps]
    return training_dataset  # + ns.white(full_config['noise_level', num_training_points)


if __name__ == "__main__":
    print("run")
    dataset = prepare_data()
    plot.plot(dataset, full_config['x_start'], label="dataset")
    params = tr.train_from_y_values(dataset)
    # params = np.random.rand(3)
    prediction = tr.iterative_forecast(params, dataset)
    plot.plot(prediction, full_config['x_start'], future_steps=full_config['future_steps'], label="prediction")
