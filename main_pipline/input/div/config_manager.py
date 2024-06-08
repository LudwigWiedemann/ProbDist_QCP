import numpy as np
import filemanager as file
import dill

time_frame_start= None,  # start of timeframe
time_frame_end = None,  # end of timeframe, needs to be bigger than time_frame_start
n_steps= None,  # How many points are in the full timeframe
time_steps= None,  # How many consecutive points are in train/test sample
future_steps= None,  # How many points are predicted in train/test sample
num_samples= None,  # How many samples of time_steps/future_steps are generated from the timeframe
noise_level= None,  # Noise level on Inputs
train_test_ratio= None,  # The higher the ratio to more data is used for training

#TODO function to load a config file
def config_load(path):

    with open(path, 'rb') as f:
        config = dill.load(f)
    time_frame_start = config['time_frame_start']
    time_frame_end = config['time_frame_end']
    n_steps = config['n_steps']
    time_steps = config['time_steps']
    future_steps = config['future_steps']
    num_samples = config['num_samples']
    noise_level = config['noise_level']
    train_test_ratio = config['train_test_ratio']
    return config


#TODO function to save a config file
def config_save():
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
    }

    with open(f"{file.path}/config.pkl", 'wb') as f:
        dill.dump(config, f)
    return