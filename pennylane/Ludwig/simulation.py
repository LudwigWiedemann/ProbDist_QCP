from pennylane import numpy as np
import noise as ns
import training as tr
import plotting as plot

x_start, x_end = 0, 20
num_training_points = 21
training_time_steps = np.linspace(x_start, x_end, num_training_points)



full_config = {
    # training data parameter
    'time_frame_start': -4 * np.pi,  # start of timeframe
    'time_frame_end': 12 * np.pi,  # end of timeframe, needs to be bigger than time_frame_start
    'n_steps': 200,  # How many points are in the full timeframe
    'time_steps': 50,  # How many consecutive points are in train/test sample
    'future_steps': 1,  # How many points are predicted in train/test sample
    'num_samples': 1000,  # How many samples of time_steps/future_steps are generated from the timeframe
    'noise_level': 0.1,  # Noise level on Inputs
    'train_test_ratio': 0.6,  # The higher the ratio to more data is used for training
    # Run parameter
    'model': 'Variable_circuit',  # PCV is the current main_model others are for baseline
    'custom_circuit': True,  # For now only relevant for PCVModel
    'circuit': 'new_RYXZ_Circuit',
    'epochs': 5,  # Adjusted to start with a reasonable number
    'batch_size': 64,  # Keep this value for now
    # Optimization parameter
    'learning_rate': 0.004,  # Adjusted to a common starting point
    'loss_function': 'mse',  # currently at 'mse'
    # Forecasting parameter
    'steps_to_predict': 300

}


def prepare_data():

    training_dataset = [tr.f(x) for x in training_time_steps]
    return training_dataset  # + ns.white(num_training_points)


if __name__ == "__main__":
    print("run")
    dataset = prepare_data()
    plot.plot(dataset)
    params = tr.train_from_y_values(dataset)
    prediction = tr.iterative_forecast(params, dataset)
    plot.plot(prediction)
