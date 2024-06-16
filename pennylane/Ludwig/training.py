import pennylane as qml
from pennylane import numpy as np
import circuit as cir
import plotting as plot

optimizer = qml.GradientDescentOptimizer(0.001)
training_iterations = 10
time_steps = 3
future_steps = 1
num_samples = 100
steps_to_forecast = 50


def f(x):
    # return np.sin(x)
    return np.sin(x) + 0.5 * np.cos(2 * x) + 0.25 * np.sin(3 * x)


def train_from_y_values(dataset):
    samples = []
    extra_sample = ()
    for s in range(num_samples):

        t_start = np.random.randint(0, len(dataset) - (time_steps + future_steps + 1))
        f_start = t_start + time_steps
        ts = dataset[t_start: t_start + time_steps]
        fs = dataset[f_start: f_start + future_steps]
        if s != num_samples - 1:
            samples.append((ts, fs))
        else:
            extra_sample = (ts, fs)

    params = np.random.rand(time_steps)
    for it in range(training_iterations):
        for sample in samples:
            params = optimizer.step(cost, params, time_steps=sample[0], expected_prediction=sample[1])

        if it % 1 == 0:
            print(f"Iteration {it}:")
            prediction = cir.multiple_wires(params, extra_sample[0])
            error = np.abs(prediction - extra_sample[1])
            print("error: " + str(error) + "average: ")
    return params


def train_params(distributions):
    param_list = [[]] * len(distributions)
    i = 0
    for dist in distributions:
        params = guess_starting_params(9)
        print("Training the circuit...")
        for iteration in range(training_iterations):
            for training_x, training_y in dist:
                params = optimizer.step(cost, params, x=training_x, target=training_y)
            total_error = 0
            if iteration % 10 == 0:
                print(f"Iteration {iteration}:")
                for training_x, training_y in dist:
                    predicted_output = cir.run_circuit(params, training_x)
                    error = np.abs(predicted_output - training_y)
                    total_error += error
                    # print( f"Input: {training_x}, Expected: {training_y:.4f}, Predicted: {predicted_output:.4f},
                    # Error: {error:.4f}")
                print("total: " + str(total_error) + "average: " + str(total_error / len(dist)))
                # plot.plot([params], [dist], f)
        param_list[i] = params
        i += 1
    return param_list


def cost(params, time_steps, expected_prediction):
    predicted_output = cir.multiple_wires(params, time_steps)
    return ((predicted_output - expected_prediction) ** 2) / 2


def guess_starting_params(total_num_params):
    print("guessing best starting parameters ... ")
    num_attempts = 3
    attempts = [[], [], []]
    errors = [99, 99, 99]
    for i in range(num_attempts - 1):
        attempts[i] = np.random.rand(total_num_params)
        x0 = 0
        x1 = np.pi
        cost_x0 = int(cost(attempts[i], x0, f(x0)))
        cost_x1 = int(cost(attempts[i], x1, f(x1)))
        mean_error = np.mean([cost_x0, cost_x1])
        errors[i] = mean_error

    best_attempt = 0
    for i in range(len(errors)):
        if errors[i] < errors[best_attempt]:
            best_attempt = i
    print("Best params: " + str(attempts[best_attempt]))
    return attempts[best_attempt]


def iterative_forecast(params, dataset):
    for i in range(steps_to_forecast):
        input = dataset[len(dataset) - time_steps:len(dataset)]
        forecast = cir.multiple_wires(params, input)
        print("forecast: " + str(forecast))
        dataset.append(forecast)
    return dataset
