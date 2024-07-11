import circuit as cir
from pennylane import numpy as np

import training

input_size = 100
np.random.seed(2)
auspraegungen = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

if __name__ == "__main__":
    input = np.random.randint(0, 10, size=input_size)
    # input = np.array([2, 2, 2, 1, 1, 1, 1, 1, 1, 2, 1, 1, 2, 2, 1, 2, 1, 1, 2, 2, 1, 2, 2, 1, 1, 2, 1, 1, 1, 2, 2, 1])
    weights = np.random.rand(cir.num_wires * 3 * cir.num_layers)
    print("\n====================================================================================================")
    print("Input Distribution:\n " + str(input))
    print("Each auspraegung (0-9):\n " + str(training.count_auspraegungen(input)))
    print("Weights: (They are not changed at the moment!!) \n" + str(weights))
    print("====================================================================================================\n")

    # print(training.scale_prediction(cir.predict_wuerfelwurf(weights)))
    training.train_prob_dist(weights, input)


