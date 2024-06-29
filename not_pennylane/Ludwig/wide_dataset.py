import training as tr  # Assuming 'training' module is correctly imported
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
    wide_dataset = [[] for _ in range(len(training_time_steps))]  # Initialize wide_dataset with empty lists

    for i, x in enumerate(training_time_steps):
        for j in range(n_y_points):
            wide_dataset[i].append(tr.f(x) * np.random.normal(0, config['noise_level']))  # Append to the inner lists

    return wide_dataset


def generate_wide_samples_set(wide_dataset, config):
    time_steps = config['time_steps']
    future_steps = config['future_steps']

    wide_Inputs = []
    wide_Outputs = []

    for _ in range(config['num_samples']):
        start_idx = np.random.randint(0, len(wide_dataset) - time_steps - future_steps)
        wide_Inputs.append(wide_dataset[start_idx:start_idx + time_steps])
        wide_Outputs.append(wide_dataset[start_idx + time_steps:start_idx + time_steps + future_steps])
    dataset = {'input_samples': wide_Inputs, 'output_samples': wide_Outputs}
    return dataset


if __name__ == '__main__':
    n_y_points = 5
    dataset = generate_wide_dataset(n_y_points, full_config)
    sample_set = generate_wide_samples_set(dataset, full_config)
    for i in range(full_config['num_samples']):
        print(f'Sample {i}:')
        input_samples = sample_set['input_samples'][i]

        output_samples = sample_set['output_samples'][i]
        for inputs in input_samples:  # Ensure you loop only till the length of the samples
            print(f'Input_sample: {inputs}')
        for outputs in output_samples:
            print(f'Output_sample: {outputs}')
