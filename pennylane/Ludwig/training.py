import pennylane as qml
from pennylane import numpy as np
import circuit as cir
from simulation import full_config as conf
import plotting as plot


training_iterations = conf['epochs']
time_steps = conf['time_steps']
future_steps = conf['future_steps']
num_samples = conf['num_samples']
total_steps_to_forecast = conf['steps_to_forecast']
learning_rate = conf['learning_rate']

optimizer = qml.GradientDescentOptimizer(learning_rate)

def f(x):
    #return np.sin(x)
    return np.sin(x) + 0.5 * np.cos(2 * x) + 0.25 * np.sin(3 * x)


def train_from_y_values(dataset):
    samples = []
    extra_sample = ()
    for s in range(num_samples):

        t_start = np.random.randint(0, len(dataset) - (time_steps + future_steps))
        f_start = t_start + time_steps
        ts = dataset[t_start: t_start + time_steps]
        fs = dataset[f_start: f_start + future_steps]
        if s != num_samples - 1:  # TODO: generate more test data sets
            samples.append((ts, fs))
        else:
            extra_sample = (ts, fs)

    # params = np.random.rand(time_steps)
    params = guess_starting_params(samples[0])
    for it in range(training_iterations):
        for sample in samples:
            params = optimizer.step(cost, params, time_steps=sample[0], expected_predictions=sample[1])

        if it % 1 == 0:
            print(f"Iteration {it}:")
            # prediction = cir.multiple_wires(params, extra_sample[0])
            error = cost(params, extra_sample[0], extra_sample[1])  # TODO: make error evaluation better with more test samples
            print("error: " + str(error) + "average: ")
    return params

#
# def train_params(distributions):
#     param_list = [[]] * len(distributions)
#     i = 0
#     for dist in distributions:
#         params = guess_starting_params(9)
#         print("Training the circuit...")
#         for iteration in range(training_iterations):
#             for training_x, training_y in dist:
#                 params = optimizer.step(cost, params, x=training_x, target=training_y)
#             total_error = 0
#             if iteration % 10 == 0:
#                 print(f"Iteration {iteration}:")
#                 for training_x, training_y in dist:
#                     predicted_output = cir.run_circuit(params, training_x)
#                     error = np.abs(predicted_output - training_y)
#                     total_error += error
#                     # print( f"Input: {training_x}, Expected: {training_y:.4f}, Predicted: {predicted_output:.4f},
#                     # Error: {error:.4f}")
#                 print("total: " + str(total_error) + "average: " + str(total_error / len(dist)))
#                 # plot.plot([params], [dist], f)
#         param_list[i] = params
#         i += 1
#     return param_list


def cost(params, time_steps, expected_predictions):
    predicted_output = cir.multiple_wires(params, time_steps)
    cost = 0
    for i in range(len(predicted_output)):
        cost += ((predicted_output[i] - expected_predictions[i]) ** 2) / 2

    return cost


def guess_starting_params(sample):
    num_weights = conf['num_weights']
    print("guessing best starting parameters ... ")
    num_attempts = 30
    attempts = [None] * num_attempts
    errors = [None] * num_attempts
    for i in range(num_attempts):
        attempts[i] = np.random.rand(num_weights)
        cost_x0 = int(cost(attempts[i], sample[0], sample[1]))
        errors[i] = cost_x0

    best_attempt = 0
    for i in range(len(errors)):
        if errors[i] < errors[best_attempt]:
            best_attempt = i
    print("Best params: " + str(attempts[best_attempt]))
    return attempts[best_attempt]


def iterative_forecast(params, dataset):
    for i in range(total_steps_to_forecast // future_steps):
        input = dataset[len(dataset) - time_steps:len(dataset)]
        forecast = cir.multiple_wires(params, input)
        print("forecast: " + str(forecast))
        print("dataset: " + str(len(dataset)))
        for elem in forecast:
            dataset.append(elem)
        plot.plot(dataset, conf['x_start'], label= f"training {i+1}")
    return dataset
