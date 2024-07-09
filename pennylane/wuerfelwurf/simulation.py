import circuit as cir
from pennylane import numpy as np

import training

input_size = 3000

if __name__ == "__main__":
    input = np.random.randint(1, 3, size=input_size)
    n = len(input)
    weights = np.random.rand(cir.num_wires * 3 * cir.num_layers)

    # print(training.scale_prediction(cir.predict_wuerfelwurf(weights)))
    training.train_prob_dist(weights, input)


