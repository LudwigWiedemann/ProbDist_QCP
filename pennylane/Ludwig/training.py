import pennylane as qml
from pennylane import numpy as np
import circuit as cir
import plotting as plot

optimizer = qml.GradientDescentOptimizer(0.001)
training_iterations = 100


def f(x):
    return np.sin(x)
    #  return np.sin(x) + 0.5*np.cos(2*x) + 0.25 * np.sin(3*x)


def train_params(training_data):
    params = guess_starting_params(9)
    print("Training the circuit...")
    for iteration in range(training_iterations):
        for training_x, training_y in training_data:
            params = optimizer.step(cost, params, x=training_x, target=training_y)
        total_error = 0
        if iteration % 10 == 0:
            print(f"Iteration {iteration}:")
            for training_x, training_y in training_data:
                predicted_output = cir.run_circuit(params, training_x)
                error = np.abs(predicted_output - training_y)
                total_error += error
                # print( f"Input: {training_x}, Expected: {training_y:.4f}, Predicted: {predicted_output:.4f},
                # Error: {error:.4f}")
            print("total: " + str(total_error) + "average: " + str(total_error / len(training_data)))
            plot.plot(params, training_data, f)
    return params


def cost(params, x, target):
    predicted_output = cir.run_circuit(params, x)
    return ((predicted_output - target) ** 2) / 2


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
