import pennylane as qml
import numpy as np
import training as tr
num_shots = 10000000
device = qml.device("default.qubit", wires=1)
device_with_shots = qml.device("default.qubit", wires=1, shots=num_shots)
num_layers = 1
num_params = 3


@qml.qnode(device)
def run_without_shots(weights, x):
    # qml.Hadamard(wires=0)
    # qml.PauliX(wires=0)

    for i in range(num_layers):
        qml.RY(x, wires=0)
        qml.RX(weights[0], wires=0)
        qml.RY(weights[1], wires=0)
        qml.RZ(weights[2], wires=0)


    probs = qml.probs(wires=0)
    eval = qml.expval(qml.PauliZ(wires=0))
    return (probs, eval)


@qml.qnode(device_with_shots)
def run_with_shots(weights, x):
    for i in range(num_layers):
        qml.RY(x, wires=0)
        qml.RX(weights[0], wires=0)
        qml.RY(weights[1], wires=0)
        qml.RZ(weights[2], wires=0)
    # qml.Hadamard(wires=0)

    probs = qml.probs(wires=0)
    eval = qml.expval(qml.PauliZ(wires=0))
    sample = qml.sample(qml.PauliZ(wires=0))
    # sample1 = qml.sample(qml.PauliZ(wires=0))
    # sample2 = qml.sample(qml.PauliZ(wires=0))
    # sample3 = qml.sample(qml.PauliZ(wires=0))
    # return sample1, sample2, sample3
    return probs, eval, sample


x = 0
# params = tr.random_new_params(num_params)
params = [0.34699692, 0.65395101, 0.46083371]


print("========================================================================================================================")
print("No Shots (always the same result for same params):")
for i in range(1):
    print("probs: " + str(run_without_shots(params, x)[0]))
    print("eval: " + str(run_without_shots(params, x)[1]))

print("========================================================================================================================")
print("With " + str(num_shots) + " Shots (result changes with same params - even if we use the same layer):")

# probs_range = []
# eval_range = []
# sample_range = []
for i in range(1):
#     print("=========================================")
#     probs = str(run_with_shots(params, x)[0])
#     probs_range.append(probs)
#     print("probs: " + str(probs))
#
    eval = str(run_with_shots(params, x)[1])
#     eval_range.append(eval)
    print("eval: " + str(eval))

#
    sample = str(run_with_shots(params, x)[2])
#     sample_range.append(sample)
    print("sample " + str(i) + ": " + str(sample))

    print("=========================================")
# print("========================================================================================================================")
# unique_probs = set(probs_range)
# unique_eval = set(eval_range)
# unique_sample = set(sample_range)
# print("probs range: " + str(unique_probs))
# print("eval range: " + str(unique_eval))
# print("sample range: " + str(unique_sample))

