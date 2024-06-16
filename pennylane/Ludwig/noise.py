from pennylane import numpy as np


def white(noise_lvl, num_training_points):
    return np.random.normal(0, noise_lvl, num_training_points)
