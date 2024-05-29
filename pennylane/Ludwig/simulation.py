from pennylane import numpy as np
import noise as ns
import training as tr
import plotting as plot

num_training_points = 60
training_inputs = np.linspace(0, 10, num_training_points)


def prepare_data(num_shots):
    training_datasets = [[]] * num_shots
    for i in range(num_shots):
        training_datasets[i] = [(x, tr.f(x)) for x in training_inputs]
        noise = ns.white(num_training_points)
        j = 0
        for x, y in training_datasets[i]:
            y += noise[j]
            j += 1
    return training_datasets


def run(num_shots):
    print("run")
    training_distributions = prepare_data(num_shots)
    param_list = tr.train_params(training_distributions)
    plot.plot(param_list, training_distributions, tr.f)


run(10)
