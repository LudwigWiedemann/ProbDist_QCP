import random

import circuit as cir
import simulation as sim
import pennylane as qml
from pennylane import numpy as np
import distribution_calculator as dc

optimizer = qml.GradientDescentOptimizer(0.01)
epochs = 200000

def train_prob_dist(weights, input):
    input_distr = count_auspraegungen(input)
    old_cost=float("inf")
    for i in range(epochs):
        old_weights = weights.copy()  # Copy of the current weights

        # Generate valid integer indices within the range of weights' indices
        random_plus = random.randint(0, len(weights) - 1)
        random_minus = random.randint(0, len(weights) - 1)

        random_value = random.uniform(0, 1)
        new_weights = weights.copy()
        new_weights[random_plus] += random_value
        new_weights[random_minus] -= random_value
        new_cost, new_pred = cost(new_weights, input)
        #print("old cost: " + str(old_cost) + " new cost: " + str(new_cost))
        if new_cost < old_cost:
            weights = new_weights  # Update weights only if new cost is lower
            current_cost = new_cost
        else:
            current_cost = old_cost
            weights= old_weights
        if current_cost == 0:
            break
        old_cost= current_cost
        if i%1==0:
            print("iteration: " + str(i))
            print("weights " + str(weights))
            print("difference " + str(input_distr) + "  " + str(new_pred))
            print("Epoch: " + str(i) + " Cost: " + str(current_cost))
    print("Final cost: " + str(current_cost))
    return weights


def distribution(weights, n):
    prediction_dist = []
    for i in range(n):
        prediction_dist.append(scale_prediction(cir.interpret_measurement(cir.predict_wuerfelwurf(weights))))
    return prediction_dist


def cost(weights, distr_in):
    count_in = count_auspraegungen(distr_in)

    prediction_dist = distribution(weights, len(distr_in))
    count_pred = count_auspraegungen(prediction_dist)
    #print("goal: " + str(count_in) + " prediction: " + str(count_pred))
    cost,_=dc.kl_divergence(count_in, count_pred)
    #print("cost: " + str(cost))
    return cost, count_pred


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
