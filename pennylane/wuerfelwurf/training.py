import circuit as cir
import simulation as sim
import pennylane as qml
from pennylane import numpy as np
from main_pipline.models_circuits_and_piplines.piplines.predict_pipline_div.distribution_calculator import kl_divergence

optimizer = qml.GradientDescentOptimizer(100)
epochs = 2000


def train_prob_dist(weights, input):
    i=0
    old_cost = cost(weights, input)
    optimizer = qml.GradientDescentOptimizer(0.01)
    while True:
    #for i in range(epochs):
        # print("iteration: " + str(i))
        # random_adjustment = np.random.normal(loc=0, scale=0.01, size=weights.shape)
        # new_weights = weights + random_adjustment
        # new_cost = cost(new_weights, input)
        # old_cost = cost(weights, input)
        # if new_cost < old_cost:
        #     weights = new_weights
        #     print("new_cost: " + str(new_cost))
        print("iteration: " + str(i))
        new_weights = optimizer.step(lambda v: cost(v, input), weights)
        current_cost = cost(weights, input)
        print(f"Iteration: {i+1}, Current Cost: {current_cost}")

        if current_cost < old_cost:
            old_cost = current_cost
            weights = new_weights
        if current_cost < 0.5:
            break
    return weights


def distribution(weights, n):
    prediction_dist = []
    for i in range(n):
        prediction_dist.append(scale_prediction(cir.predict_wuerfelwurf(weights)))
    return prediction_dist


def cost(weights, distr_in):
    count_in = normalize_counts(count_auspraegungen(distr_in))
    prediction_dist = distribution(weights, len(distr_in))
    cost = 0.0  # Initialize cost as a float
    count_pred = normalize_counts(count_auspraegungen(prediction_dist))
    print("goal: " + str(count_in) + " prediction: " + str(count_pred))
    # Assuming kl_divergence returns a float; ensure it's not implicitly converting to int
    cost,_ = kl_divergence(count_in, count_pred)
    return cost  # This ensures cost is returned as a float

import numpy as np

def normalize_counts(counts):
    total = np.sum(counts)
    if total > 0:
        normalized_counts = counts / total
    else:
        # Explicitly return an array of zeros with the same shape as counts
        normalized_counts = np.zeros_like(counts)
    return normalized_counts

def scale_prediction(pred):
    # print(pred)
    # return pred
    if -1 <= pred < -0.8:
        return np.array(0)
    if -0.8 <= pred < -0.6:
        return np.array(1)
    if -0.6 <= pred < -0.4:
        return np.array(2)
    if -0.4 <= pred < -0.2:
        return np.array(3)
    if -0.2 <= pred < 0.0:
        return np.array(4)
    if 0.0 <= pred < 0.2:
        return np.array(5)
    if 0.2 <= pred < 0.4:
        return np.array(6)
    if 0.4 <= pred < 0.6:
        return np.array(7)
    if 0.6 <= pred < 0.8:
        return np.array(8)
    if 0.8 <= pred < 1.0:
        return np.array(9)


def count_auspraegungen(distr):
    count = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    for elem in distr:
        if elem == 0:
            count[0] += 1
        if elem == 1:
            count[1] += 1
        if elem == 2:
            count[2] += 1
        if elem == 3:
            count[3] += 1
        if elem == 4:
            count[4] += 1
        if elem == 5:
            count[5] += 1
        if elem == 6:
            count[6] += 1
        if elem == 7:
            count[7] += 1
        if elem == 8:
            count[8] += 1
        if elem == 9:
            count[9] += 1

    return count
