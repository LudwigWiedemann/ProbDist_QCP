import circuit as cir
from pennylane import numpy as np

import training

import warnings

# Suppress specific warnings from autograd
warnings.filterwarnings("ignore", message="Output seems independent of input.")

input_size = 3000

if __name__ == "__main__":
    input = np.random.randint(1, 3, size=input_size)
    n = len(input)
    weights = np.random.rand(cir.num_wires * 3 * cir.num_layers)

    print("trainingscale: "+str(training.scale_prediction(cir.predict_wuerfelwurf(weights))))
    print("training_prob_distr: "+str(training.train_prob_dist(weights, input)))


