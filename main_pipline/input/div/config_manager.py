import numpy as np
import main_pipline.input.div.filemanager as file
import dill

time_frame_start = None  # start of timeframe
time_frame_end = None  # end of timeframe, needs to be bigger than time_frame_start
n_steps = None  # How many points are in the full timeframe
time_steps = None  # How many consecutive points are in train/test sample
future_steps = None  # How many points are predicted in train/test sample
num_samples = None  # How many samples of time_steps/future_steps are generated from the timeframe
noise_level = None  # Noise level on Inputs
train_test_ratio = None  # The higher the ratio to more data is used for training
# run Parameters
model = None  # PCV is the current main_model others are for baseline
custom_circuit = True  # For now only relevant for PCVModel
circuit = None
epochs = None  # Adjusted to start with a reasonable number
batch_size = None  # Keep this value for now
# Optimization parameter
learning_rate = None  # Adjusted to a common starting point
loss_function = None,  # currently at 'mse'
# Forecasting parameter
steps_to_predict = None


def config_load(path):
    """
    Load the config from the file
    :param path: str: path to the config file
    :return: list: config
    """
    global time_frame_start, time_frame_end, n_steps, time_steps, future_steps, num_samples, noise_level, train_test_ratio, model, custom_circuit, circuit, epochs, batch_size, learning_rate, loss_function, steps_to_predict
    with open(path, 'rb') as f:
        config = dill.load(f)
    time_frame_start = int(config['time_frame_start'])
    time_frame_end = int(config['time_frame_end'])
    n_steps = int(config['n_steps'])
    time_steps = int(config['time_steps'])
    future_steps = int(config['future_steps'])
    num_samples = int(config['num_samples'])
    noise_level = int(config['noise_level'])
    train_test_ratio = int(config['train_test_ratio'])
    model = config['model']
    custom_circuit = bool(config['custom_circuit'])
    circuit = config['circuit']
    epochs = int(config['epochs'])
    batch_size = int(config['batch_size'])
    learning_rate = int(config['learning_rate'])
    loss_function = config['loss_function']
    steps_to_predict = int(config['steps_to_predict'])
    return config


def load_from_values(config):
    """
    Load the config from input
    :param config: string: configname
    :return: none
    """
    global time_frame_start, time_frame_end, n_steps, time_steps, future_steps, num_samples, noise_level, train_test_ratio, model, custom_circuit, circuit, epochs, batch_size, learning_rate, loss_function, steps_to_predict
    time_frame_start = int(config[0])
    time_frame_end = int(config[1])
    n_steps = int(config[2])
    time_steps = int(config[3])
    future_steps = int(config[4])
    num_samples = int(config[5])
    noise_level = int(config[6])
    train_test_ratio = int(config[7])
    model = config[8]
    custom_circuit = bool(config[9])
    circuit = config[10]
    epochs = int(config[11])
    batch_size = int(config[12])
    learning_rate = int(config[13])
    loss_function = config[14]
    steps_to_predict = int(config[15])


def config_save():
    """
    Save the config to the file
    :return: None
    """
    file.create_folder()
    config = {
        'time_frame_start': time_frame_start,
        'time_frame_end': time_frame_end,
        'n_steps': n_steps,
        'time_steps': time_steps,
        'future_steps': future_steps,
        'num_samples': num_samples,
        'noise_level': noise_level,
        'train_test_ratio': train_test_ratio,
        'model': model,
        'custom_circuit': custom_circuit,
        'circuit': circuit,
        'epochs': epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'loss_function': loss_function,
        'steps_to_predict': steps_to_predict
    }

    with open(f"{file.path}/config.pkl", 'wb') as f:
        dill.dump(config, f)


def config_create(config):
    """
    Create the config from the input
    :param config: str:  configname
    :return: list: config
    """
    return {'time_frame_start': int(config[0]),
            'time_frame_end': int(config[1]),
            'n_steps': int(config[2]),
            'time_steps': int(config[3]),
            'future_steps': int(config[4]),
            'num_samples': int(config[5]),
            'noise_level': int(config[6]),
            'train_test_ratio': int(config[7]),
            'model': config[8],
            'custom_circuit': bool(config[9]),
            'circuit': config[10],
            'epochs': int(config[11]),
            'batch_size': int(config[12]),
            'learning_rate': int(config[13]),
            'loss_function': config[14],
            'steps_to_predict': int(config[15])
            }


def to_list(config):
    """
    Convert the config to a list
    :param config: str: configname
    :return: list: config
    """
    return [config['time_frame_start'], config['time_frame_end'], config['n_steps'], config['time_steps'],
            config['future_steps'], config['num_samples'], config['noise_level'], config['train_test_ratio']]
