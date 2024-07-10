import circuit
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
    prediction_dist = distribution(weights, len(distr_in))
    cost = 0
    count_in = count_auspraegungen(distr_in)
    count_pred = count_auspraegungen(prediction_dist)
    i=0
    print(str(i)+": goal: " + str(count_in) + " prediction: " + str(count_pred))
    for auspr in [0, 1]:
        cost += ((count_in[auspr] - count_pred[auspr]) ** 2)
        i+=1
    print("cost: " + str(cost))
    return cost


def scale_prediction(pred):
    return pred
    # if pred < 0:
    #     return 1
    # else:
    #     return 2


def count_auspraegungen(distr):
    count = np.array([0, 0])
    for elem in distr:
        if elem < 0:
            count[0] += 1
        if elem >= 0:
            count[1] += 1
    return count
