import numpy as np
import dill
from random import sample

from main_pipline.models_circuits_and_piplines.piplines.predict_pipline_div.predict_plots_and_metrics import \
    plot_full_timeframe_data, show_sample_preview_plots

# TODO write all relevant settings. Currently only examples
dataset_settings = {
    "name": "Test_set_4_150_1_15",
    "n_wires": 4,  # Number of wires for the quantum device.
    "min_target_depth": 5,  # Minimum depth of target.
    "max_target_depth": 15,  # Maximum depth of target.
    "size": 150
}

test_config = {
    # Example training data parameters

    'time_frame_start': -4 * np.pi,  # start of timeframe
    'time_frame_end': 12 * np.pi,  # end of timeframe, needs to be bigger than time_frame_start
    'n_steps': 200,  # How many points are in the full timeframe
    'time_steps': 50,  # How many consecutive points are in train/test sample
    'future_steps': 10,  # How many points are predicted in train/test sample
    'num_samples': 1000,  # How many samples of time_steps/future_steps are generated from the timeframe
    'noise_level': 0.1,  # Noise level on Inputs
    'train_test_ratio': 0.6,  # The higher the ratio to more data is used for training
}


# TODO should function also be stored so we know which function dataset was used for?
def function(x):
    """
    Function to get y values for x values. Add function into the dictionary and call it in return statement.
    :param x: x value
    :return: y value
    """
    function_dictionary = {
        "sin": np.sin(x),
        "cos": np.cos(x),
        "complex_function": np.sin(x) + 0.5 * np.cos(2 * x) + 0.25 * np.sin(3 * x)
    }
    return function_dictionary.get("sin")


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
    noisy_y_data = y_data + np.random.normal(0, noise_level, size=y_data.shape)

    input_train = []
    output_train = []

    input_noisy_test = []
    output_test = []
    input_real_test = []

    if time_steps + future_steps >= len(x_data):
        raise ValueError("time_steps + future_steps must be less than the length of x_data")

    preview_sample = sample(range(num_samples), config['preview_samples'])
    for i in range(num_samples):
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
        if i in preview_sample:
            show_sample_preview_plots(input_sample, output_sample, input_noisy_sample, config)

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
    plot_full_timeframe_data(x_data, y_data, noisy_y_data, title='Full Timeframe Data', show=config['show_dataset_plots'])
    dataset = {'input_train': input_train, 'output_train': output_train, 'input_noisy_test': input_noisy_test,
               'input_test': input_real_test, 'output_test': output_test,
               'input_forecast': input_foresight, 'input_noisy_forecast': input_noisy_foresight, 'step_size': step_size}

    return dataset


def generate_and_save_dataset(dataset_name, config):
    """
    Generates the dataset with given config and saves it into a deserializable pickle
    file in the datasets directory
    """
    dataset_to_save = generate_time_series_data(function, config)
    with open(f"datasets/{dataset_name}.pkl", 'wb') as f:
        dill.dump(dataset_to_save, f)


def load_dataset(dataset_name):
    """
    Loads the given pickle file from datasets directory
    """
    with open(f"datasets/{dataset_name}.pkl", 'rb') as f:
        dataset_to_load = dill.load(f)
    return dataset_to_load


def test_functionality(file_name="test_dataset"):
    """
    Test if the save load works with a valid config
    """
    generate_and_save_dataset(file_name, test_config)
    dataset = load_dataset(file_name)
    print(dataset)


if __name__ == '__main__':
    # generate_and_save_dataset("PAT", test_config)
    test_functionality()
    # test_functionality("WOWXDGHG")
