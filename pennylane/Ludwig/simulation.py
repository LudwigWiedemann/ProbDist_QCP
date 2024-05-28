from pennylane import numpy as np
import noise as ns
import training as tr
import plotting as plot

num_training_points = 60
training_inputs = np.linspace(0, 10, num_training_points)


def prepare_data():
    training_data = [(x, tr.f(x)) for x in training_inputs]
    noise = ns.white(num_training_points)
    i = 0
    for x, y in training_data:
        y += noise[i]
        i += 1
    return training_data


def run():
    print("run")
    training_data = prepare_data()
    optimized_params = tr.train_params(training_data)
    plot.plot(optimized_params, training_data, tr.f)


run()
