import numpy as np

from main_pipline.models_circuits_and_piplines.piplines.predict_pipline_div.predict_plots_and_metrics import \
    plot_full_timeframe_data

# TODO write all relevant settings. Currently only examples
dataset_settings = {
    "name": "Test_set_4_150_1_15",
    "n_wires": 4,  # Number of wires for the quantum device.
    "min_target_depth": 5,  # Minimum depth of target.
    "max_target_depth": 15,  # Maximum depth of target.
    "size": 150
}


# TODO expand list of potential functions
def function():
    return np.sin()
# NOT USED ATM


def generate_time_series_data(func, config):
    time_steps = config['time_steps']
    num_samples = config['num_samples']
    future_steps = config['future_steps']
    noise_level = config.get('noise_level', 0.0)
    n_steps = config['n_steps']

    # Ensure the length of x_data is greater than time_steps
    n_steps = max(time_steps + future_steps, n_steps)  # Ensure a minimum length
    x_data = np.linspace(config['time_frame_start'], config['time_frame_end'],
                         n_steps)
    y_data = func(x_data)
    noisy_y_data = y_data + np.random.normal(0, noise_level, size= y_data.shape)

    input_train = []
    output_train = []

    input_noisy_test = []
    output_test = []
    input_real_test = []

    if time_steps + future_steps >= len(x_data):
        raise ValueError("time_steps + future_steps must be less than the length of x_data")

    for _ in range(num_samples):
        start_idx = np.random.randint(0, len(x_data) - time_steps - future_steps)
        input_sample = y_data[start_idx:start_idx + time_steps]
        input_noisy_sample = noisy_y_data[start_idx:start_idx + time_steps]
        output_sample = y_data[start_idx + time_steps:start_idx + time_steps + future_steps]



        if np.random.rand() < config['train_test_ratio']:
            input_train.append(input_noisy_sample)
            output_train.append(output_sample)
        else:
            input_noisy_test.append(input_noisy_sample)
            input_real_test.append(input_sample)
            output_test.append(output_sample)



    # Prepare data for foresight
    future_start_idx = len(x_data) - time_steps
    input_foresight = y_data[future_start_idx:future_start_idx + time_steps]
    input_noisy_foresight = input_foresight + np.random.normal(0, noise_level, input_foresight.shape)
    step_size = (config['time_frame_end'] - config['time_frame_start']) / (n_steps - 1)

    input_train = np.array(input_train).reshape(-1, time_steps, 1)
    output_train = np.array(output_train).reshape(-1, future_steps)
    input_real_test = np.array(input_real_test).reshape(-1, time_steps, 1)
    input_noisy_test = np.array(input_noisy_test).reshape(-1, time_steps, 1)
    output_test = np.array(output_test).reshape(-1, future_steps)
    input_foresight = np.array(input_foresight).reshape(1, time_steps, 1)
    input_noisy_foresight = np.array(input_noisy_foresight).reshape(1, time_steps, 1)

    plot_full_timeframe_data(x_data, y_data,noisy_y_data, title='Full Timeframe Data')

    dataset = {'input_train': input_train, 'output_train': output_train, 'input_noisy_test': input_noisy_test,
               'input_test': input_real_test, 'output_test': output_test,
               'input_forecast': input_foresight, 'input_noisy_forecast': input_noisy_foresight,'step_size': step_size}

    return dataset


# TODO function to save a dataset as a file use dill lib preferable
def generate_and_save_dataset(save_path, config):
    return


# TODO function to load a dataset from a file
def load_dataset(load_path, dataset, name):
    return
