import time
from datetime import datetime

import pennylane as qml
from pennylane import numpy as np
import circuit as cir
import simulation as sim
from simulation import full_config as conf
import plotting as plt


training_iterations = conf['epochs']
time_steps = conf['time_steps']
future_steps = conf['future_steps']
num_samples = conf['num_samples']
total_steps_to_forecast = conf['steps_to_forecast']
learning_rate = conf['learning_rate']

# optimizer = qml.GradientDescentOptimizer(learning_rate)
optimizer = qml.AdamOptimizer(learning_rate)

def f(x):
    #return np.sin(x)
    # return np.sin(x) + 0.5 * np.cos(2 * x) + 0.25 * np.sin(3 * x)
    return np.sin(x) - 0.5 * np.cos(2 * x) + 0.25 * np.sin(3 * x)



def train_from_y_values(dataset):
    training_samples = []
    test_samples = []
    # for s in range(len(dataset) -(time_steps + future_steps + 1)):
    for s in range(num_samples):
        # t_start = 0
        t_start = np.random.randint(0, len(dataset) - (time_steps + future_steps))
        # t_start = s

        f_start = t_start + time_steps
        ts = dataset[t_start: t_start + time_steps]
        fs = dataset[f_start: f_start + future_steps]
        if s % 20 == 0:  # TODO: generate more test data sets
            test_samples.append((ts, fs))
            plt.plot_sample((ts, fs), sim.step_size, s)
        else:
            training_samples.append((ts, fs))

    # params = np.random.rand(time_steps)
    first_sample = training_samples[0]
    middle_sample = training_samples[len(training_samples) // 2]
    last_sample = training_samples[len(training_samples)-1]
    # params = guess_starting_params(first_sample, middle_sample, last_sample)
    params = np.array([np.random.rand(sim.num_weights) for _ in range(sim.num_weights_per_layer)])
    start_time = datetime.now()
    iteration_start_time = datetime.now()
    for it in range(training_iterations):
        if it % 10 == 1:
            iteration_start_time = datetime.now()
        batch_size = 20
        for start in range(int(len(training_samples)/batch_size)):
            cost_samples = training_samples[start*batch_size:(start+1)*batch_size]
            # gradients = tf.gradients(cost(params,sample[0], sample[1]), params)
            # params = optimizer.step(gradients, params, time_steps=sample[0], expected_predictions=sample[1])

            params = optimizer.step(cost, params, samples=cost_samples)

        if it % 10 == 0:
            total_error = 0
            # for test_sample in test_samples:
            #     total_error += cost(params, test_sample)
            # prediction = cir.multiple_wires(params, extra_sample[0])
            print("average error of all test samples: " + str(total_error/len(test_samples)))
            end_time = datetime.now()
            elapsed_time = end_time - start_time
            print(f"Iteration {it}: {datetime.now() - iteration_start_time}")
            start_time = datetime.now()
    return params


def cost(params, samples):
    total_cost = 0
    for sample in samples:
        inputs = sample[0]
        compare_outputs = sample[1]
        predicted_output = cir.multiple_wires(params, inputs)
        sample_cost = 0
        for i in range(len(predicted_output)):
            po = predicted_output[i]
            co = compare_outputs[i]
            bla = ((po - co) ** 2)
            sample_cost += bla
            # cost += np.abs((predicted_output[i] - compare_outputs[i]))
        total_cost += sample_cost
    return total_cost


def guess_starting_params(first, middle, last):
    samples = [first, middle, last]
    print("guessing best starting parameters ... ")
    num_attempts = 30
    attempts = [None] * num_attempts
    errors = [None] * num_attempts
    for i in range(num_attempts):
        attempts[i] = np.random.rand(sim.num_weights)
        cost_x0 = cost(attempts[i], samples)
        errors[i] = cost_x0

    best_attempt = 0
    for i in range(len(errors)):
        if errors[i] < errors[best_attempt]:
            best_attempt = i
    # print("Best params: " + str(attempts[best_attempt]))
    return attempts[best_attempt]


def iterative_forecast(params, dataset):
    for i in range(total_steps_to_forecast // future_steps):
        input = dataset[len(dataset) - time_steps:len(dataset)]
        forecast = cir.multiple_shots(params, input)
        # print("forecast: " + str(forecast))
        # print("dataset: " + str(len(dataset)))
        for elem in forecast:
            dataset.append(elem)

        # if i % 3 == 0:
        #     plot.plot(dataset, conf['x_start'], sim.step_size, label= f"training {i+1}")
    return dataset


def scale(n):
    return n/scale_factor
