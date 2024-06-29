import training as tr
import numpy as np


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
