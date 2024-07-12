from pennylane import numpy as np
import dill
from random import sample

from main_pipline.models_circuits_and_piplines.piplines.predict_pipline_div.predict_plots_and_metrics import \
    plot_full_timeframe_data, show_sample_preview_plots

def function0(x):
    return np.sin(x) + 0.5 * np.cos(2 * x) + 0.25 * np.sin(3 * x)
def function1(x):
    return np.sin(x) * np.cos(2 * x) + 0.5 * np.sin(2 * x) * np.cos(3 * x)

def function2(x):
    return np.sin(x) + 0.9 * np.sin(2 * x)

def function3(x):
    return np.cos(x) + 0.3 * np.cos(2 * x)

def function4(x):
    return np.cos(x / 2) + 0.5 * np.sin(3 * x**2) - 0.4 * np.cos(4 * x**0.5)

def function5(x):
    return np.cos(x) + 0.75 * np.sin(2 * x) + 0.5 * np.cos(3 * x)


def generate_dataset(func, config, logger):
    x_data, y_data, noisy_y_data = generate_timeline(config, func, logger)

    input_train, output_train, input_noisy_test, output_test, input_real_test = generate_samples(config, x_data, y_data,
                                                                                                 noisy_y_data, logger)
    # Prepare data for foresight
    input_foresight, input_noisy_foresight, extended_y_data, extended_noisy_y_data, extended_forecast_sample = generate_foresight_samples(
        config, x_data, y_data, func)

    dataset = {'input_train': input_train, 'output_train': output_train,
               'input_noisy_test': input_noisy_test, 'input_test': input_real_test, 'output_test': output_test,
               'input_forecast': input_foresight, 'input_noisy_forecast': input_noisy_foresight,
               'extended_y_data': extended_y_data, 'extended_noisy_y_data': extended_noisy_y_data,
               'extended_forecast_sample': extended_forecast_sample}

    return dataset


def generate_timeline(config, func, logger):
    time_steps = config['time_steps']
    future_steps = config['future_steps']
    noise_level = config.get('noise_level', 0.0)
    n_steps = config['n_steps']

    # Ensure the length of x_data is greater than time_steps
    n_steps = max(time_steps + future_steps, n_steps)  # Ensure a minimum length
    x_data = np.linspace(config['time_frame_start'], config['time_frame_end'], n_steps)
    y_data = func(x_data)
    noisy_y_data = y_data + np.random.normal(0, noise_level, size=y_data.shape)
    plot_full_timeframe_data(x_data, y_data, noisy_y_data, title='Full Timeframe Data',
                             show=config['show_dataset_plots'], logger=logger)
    return x_data, y_data, noisy_y_data


def generate_samples(config, x_data, y_data, noisy_y_data, logger):
    time_steps = config['time_steps']
    future_steps = config['future_steps']
    num_samples = config['num_samples']

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
        show_sample_preview_plots(input_sample, output_sample, input_noisy_sample, config, logger)

    input_train = np.array(input_train).reshape(-1, time_steps, 1)
    output_train = np.array(output_train).reshape(-1, future_steps, 1)

    input_real_test = np.array(input_real_test).reshape(-1, time_steps, 1)
    input_noisy_test = np.array(input_noisy_test).reshape(-1, time_steps, 1)
    output_test = np.array(output_test).reshape(-1, future_steps, 1)

    return input_train, output_train, input_noisy_test, output_test, input_real_test


def generate_foresight_samples(config, x_data, y_data, func):
    time_steps = config['time_steps']
    noise_level = config['noise_level']
    future_steps = config['future_steps']

    future_start_idx = len(x_data) - time_steps
    input_foresight = y_data[future_start_idx:future_start_idx + time_steps]
    input_noisy_foresight = input_foresight + np.random.normal(0, noise_level, input_foresight.shape)
    step_size = (config['time_frame_end'] - config['time_frame_start']) / (config['n_steps'] - 1)

    extended_x_data = step_size * config['steps_to_predict']
    extended_y_data = func(
        np.linspace(config['time_frame_end'], config['time_frame_end'] + extended_x_data, config['steps_to_predict']))
    extended_noisy_y_data = extended_y_data + np.random.normal(0, noise_level, size=extended_y_data.shape)

    forecast_y_data = np.concatenate([input_noisy_foresight, extended_noisy_y_data])
    extended_forecast_sample = []
    for i in range((config['steps_to_predict'] // future_steps)+1):
        foresight_input = forecast_y_data[i * future_steps: i * future_steps + time_steps]
        extended_forecast_sample.append(np.array(foresight_input).reshape(-1, time_steps, 1))

    input_foresight = np.array(input_foresight).reshape(-1, time_steps, 1)
    input_noisy_foresight = np.array(input_noisy_foresight).reshape(-1, time_steps, 1)
    return input_foresight, input_noisy_foresight, extended_y_data, extended_noisy_y_data, extended_forecast_sample



def generate_and_save_dataset(dataset_name, config):
    """
    Generates the dataset with given config and saves it into a deserializable pickle
    file in the datasets directory
    """
    dataset_to_save = generate_data(function, config)
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
    generate_and_save_dataset(file_name)
    dataset = load_dataset(file_name)
    print(dataset)


if __name__ == '__main__':
    # generate_and_save_dataset("PAT", test_config)
    test_functionality()
    # test_functionality("WOWXDGHG")
