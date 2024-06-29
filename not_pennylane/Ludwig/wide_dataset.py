import training as tr
import numpy as np

full_config = {
    # data parameter
    'x_start': -4 * np.pi,
    'x_end': 12 * np.pi,
    'total_training_points': 2000,
    'noise_level': 0.1,  # Noise level on Inputs
    'train_test_ratio': 0.6,  # The higher the ratio to more data is used for training

    # circuit parameter
    'weights_per_layer': 6,
    'num_layers': 1,

    # training parameter
    'time_steps': 8,  # How many consecutive points are in train/test sample
    'future_steps': 2,  # How many points are predicted in train/test sample
    'num_samples': 20,  # How many samples of time_steps/future_steps are generated from the timeframe
    'epochs': 20,  # Adjusted to start with a reasonable number
    'learning_rate': 0.0005,  # Adjusted to a common starting point
    # Forecasting parameter
    'steps_to_forecast': 50,
    'num_shots_for_evaluation': 200,
    'predictions_for_distribution': 10

}


def generate_wide_dataset(n_y_points, config):
    training_time_steps = np.linspace(config['x_start'], config['x_end'],
                                      config['total_training_points'])
    wide_dataset = []
    for i, x in enumerate(training_time_steps):
        wide_dataset[i] = [] * n_y_points
        for j in range(n_y_points):
            wide_dataset[i][j] = tr.f(x) * config['noise_level']
    return wide_dataset


def generate_wide_samples_set(wide_dataset, config):
    time_steps = config['time_steps']
    future_steps = config['future_steps']

    wide_Inputs = []
    wide_Outputs = []
    for _ in config['num_samples']:
        start_idx = np.random.randint(0, len(wide_dataset) - time_steps - future_steps)
        wide_Inputs.append(wide_dataset[start_idx:start_idx + time_steps])
        wide_Outputs.append(wide_dataset[start_idx + time_steps:start_idx + time_steps + future_steps])
    return {'input_samples': wide_Inputs, 'output_samples': wide_Outputs}


if __name__ == '__main__':
    dataset = generate_wide_dataset(5, full_config)
    sample_set = generate_wide_samples_set(dataset, full_config)
