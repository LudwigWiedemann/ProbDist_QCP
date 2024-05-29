import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

dev = qml.device("default.qubit", wires=0)


def training_input(start, stop, num_points):
    x = np.linspace(start, stop, num_points)
    return x


def target_function():
    return np.sin(training_input(0, 2 * np.pi, 50))


@qml.qnode(dev)
def circuit(params):
    qml.RY(params[0], wires=0)
    return qml.expval(qml.PauliZ(0))


def loss_function_mse(prediction, target):
    return np.mean((prediction - target) ** 2)


def cost(params, target):
    prediction = circuit(params)
    return loss_function_mse(prediction, target)


if __name__ == '__main__':
    params = np.random.rand()
    params = qml.numpy.array(params, requires_grad=True)
    opt = qml.AdamOptimizer(0.1)
    steps = 100

    for i in range(steps):
        opt.step(lambda params: cost(params, target_function()), params)
        #opt.step(cost(params, target_function()))
        print(f"Cost after step {i + 1}/{steps}: {cost(params, target_function())}")
    # print(f"Optimized parameter: {params}")
    # print(f"Target function value: {target_function()}")
    # print(f"Predicted value: {circuit(params)}")
    # print(f"Error: {np.abs(target_function() - circuit(params))}")

    x = np.linspace(0, 2 * np.pi, 100)
    y = target_function()
    y_pred = [circuit(params) for _ in x]
    plt.plot(y, label="Target function")
    plt.plot(y_pred, label="Predicted function")
    plt.legend()
    plt.show()
