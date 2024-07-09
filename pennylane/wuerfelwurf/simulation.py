import circuit as cir
from pennylane import numpy as np

import training

if __name__ == "__main__":
    input = [1,1,1,2]
    n = len(input)
    weights = np.random.rand(cir.num_wires * 3 * cir.num_layers)

    # print(training.scale_prediction(cir.predict_wuerfelwurf(weights)))
    training.train_prob_dist(weights, input)


