import circuit
import simulation as sim
import pennylane as qml
from pennylane import numpy as np

optimizer = qml.GradientDescentOptimizer(0.01)
epochs = 20


def train_prob_dist(weights, input):
    for i in range(epochs):
        weights = optimizer.step(cost, weights, distr_in=input)
    return weights


def distribution(weights, n):
    prediction_dist = []
    for i in range(n):
        prediction_dist.append(scale_prediction(circuit.predict_wuerfelwurf(weights)))
    return prediction_dist


def cost(weights, distr_in):
    count_in = count_auspraegungen(distr_in)

    prediction_dist = distribution(weights, len(distr_in))
    cost = 0
    count_pred = count_auspraegungen(prediction_dist)
    print("goal: " + str(count_in) + " prediction: " + str(count_pred))
    for auspr in sim.auspraegungen:
        cost += ((count_in[auspr] - count_pred[auspr]) ** 2)
    print("cost: " + str(cost))
    return cost


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
