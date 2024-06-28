import math

from pennylane import numpy as np
import training as tr
import plotting as plot
import distribution_calculator as dc
from datetime import datetime


full_config = {
    # data parameter
    'x_start': 0,
    'x_end': 10,
    'total_training_points': 50,
    'noise_level': 0.1,  # Noise level on Inputs
    'train_test_ratio': 0.6,  # The higher the ratio to more data is used for training

    # circuit parameter
    'weights_per_layer': 3,
    'num_layers': 1,

    # training parameter
    'time_steps': 8,  # How many consecutive points are in train/test sample
    'future_steps': 2,  # How many points are predicted in train/test sample
    'num_samples': 80,  # How many samples of time_steps/future_steps are generated from the timeframe
    'epochs': 50,  # Adjusted to start with a reasonable number
    'learning_rate': 0.01,  # Adjusted to a common starting point
    # Forecasting parameter
    'steps_to_forecast': 100,
    'num_shots_for_evaluation': 200,
    'predictions_for_distribution': 10

}
step_size = ((full_config['x_end'] - full_config['x_start']) / (full_config['total_training_points'] - 1))
num_weights = full_config['time_steps'] * full_config['weights_per_layer'] * full_config['num_layers']
num_wires = int(math.log2(full_config['time_steps']))

def prepare_data():
    training_time_steps = np.linspace(full_config['x_start'], full_config['x_end'],
                                      full_config['total_training_points'])
    training_dataset = [tr.f(x) for x in training_time_steps]
    return training_dataset  # + ns.white(full_config['noise_level', num_training_points)

def prepare_extended_data():
    training_time_steps = np.linspace(full_config['x_start'],
                                      full_config['x_end']+full_config['steps_to_forecast'],
                                      full_config['total_training_points']+full_config['steps_to_forecast']
                                      )
    training_dataset = [tr.f(x) for x in training_time_steps]
    return training_dataset  # + ns.white(full_config['noise_level', num_training_points)


if __name__ == "__main__":
    print("run")
    for i in range(1):
        dataset = prepare_data()
        # plot.plot(dataset, full_config['x_start'], step_size, full_config['total_training_points'])
        plot.plot(dataset, full_config['x_start'], step_size, full_config['total_training_points'])
        params = tr.train_from_y_values(dataset)
        # params = np.random.rand(3)
        # dataset = dataset[0:full_config['time_steps']]
        predictions = []
        prediction_start_time = datetime.now()
        probabilities=[]
        #shots
        for i in range(full_config['predictions_for_distribution']):
            prediction_dataset = list(dataset)
            prediction, prob = tr.iterative_forecast(params, prediction_dataset)
            predictions.append(prediction)
        extended_dataset= prepare_extended_data()
        dc.calculate_distribution_with_KLD(predictions, [extended_dataset], step_size, full_config['x_start'], full_config['x_end'])
        print("PREDICTIONS:!!!")
        print(predictions)
        #average_divergent=dc.average_kl_divergence(probabilities)
        #plot.plot_kl_divergence(average_divergent)
        prediction_end_time = datetime.now()
        print("prediction took", prediction_end_time - prediction_start_time)
        plot.plot_evaluation(predictions, full_config['x_start'], step_size, full_config['total_training_points'])
