from datetime import datetime

import pennylane as qml
from pennylane import numpy as np
import circuit as cir

optimizer = qml.GradientDescentOptimizer(0.001)


def f(x):
    # return np.sin(x)
    return np.sin(x) + 0.5 * np.cos(2 * x) + 0.25 * np.sin(3 * x)


def random_new_params(len):
    print("guessing best starting parameters ... ")
    return np.random.rand(len)


def train_weights(weights, training_data, iterations):
    print("Training the circuit...")
    start_time = datetime.now()
    for iteration in range(iterations):
        total_error = 0
        average_error = 0

        for x, y in training_data:
            # total_error += cost(weights, x, y)
            weights = optimizer.step(cost, weights, x=x, target=y)

        if iteration % 10 == 0:
            end_time = datetime.now()
            time_for_ten_rotations = end_time - start_time
            start_time = datetime.now()
            # average_error = total_error / len(training_data)
            # evaluation = cir.run_without_shots(weights, 0)
            print(f"Iteration {iteration} duration: " + str(time_for_ten_rotations)) # + " average error: " + str(average_error)) # + " evaluation: " + str(evaluation))
    end_training = datetime.now()
    training_time = end_training - start_training
    print("Training time: " + str(training_time))
    print("Trained weights: " + str(weights))
    return weights


def cost(weights, x, target):
    prediction = cir.run_without_shots(weights, x)
    return ((prediction - target) ** 2) / 2
