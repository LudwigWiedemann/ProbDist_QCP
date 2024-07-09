import circuit
import pennylane as qml
from pennylane import numpy as np

optimizer = qml.GradientDescentOptimizer(0.01)
epochs = 2
input = [1, 1, 1, 2]


def train_prob_dist(weights, input):

    for i in range(epochs):
        print("hi")
        weights = optimizer.step(cost, weights)
        print("hi")

        # params = optimizer.step(cost, params, samples=cost_samples)

    return weights

def cost(weights):
    prediction_dist = []
    for i in range(len(input)):
        prediction_dist.append(scale_prediction(circuit.predict_wuerfelwurf(weights)))
    cost = 0
    count_in = count_auspraegungen(input)
    count_pred = count_auspraegungen(prediction_dist)
    for auspr in [0,1]:
        cost += (count_in[auspr] - count_pred[auspr]) ** 2
    return cost



def scale_prediction(pred):
    return pred
    # if pred < 0:
    #     return 1
    # else:
    #     return 2


def count_auspraegungen(distr):
    count = np.array([0,0])
    for elem in distr:
        if elem < 0:
            count[0] += 1
        if elem >= 0:
            count[1] += 1
    return count
