import pennylane as qml
import simulation as sim
from simulation import full_config as conf
import math
from pennylane import numpy as np

#
# num_qubits = 1
# num_layers = 9
# device = qml.device("default.qubit", wires=num_qubits)
# num_wires = 5
# num_outputs = 5

# num_wires = conf['time_steps']
# num_wires = 5

num_outputs = conf['future_steps']
num_wires = sim.num_wires
num_shots = conf['num_shots_for_evaluation']
num_layers = conf['num_layers']

proberbilities =[]

#
# @qml.qnode(device)
# def run_circuit(params, x):
#     qml.RY(params[0] * x, wires=0)
#     qml.RY(params[1] * x, wires=0)
#     qml.RY(params[2], wires=0)
#     qml.RY(params[3], wires=0)
#     qml.RY(params[4] * x, wires=0)
#     qml.RY(params[5], wires=0)
#     qml.RY(params[6], wires=0)
#     qml.RY(params[7], wires=0)
#     qml.RY(params[8], wires=0)
#     return qml.expval(qml.PauliZ(wires=0))


dev = qml.device("default.qubit", wires=num_wires)
shot_dev = qml.device("default.qubit", wires=num_wires, shots=num_shots)


@qml.qnode(dev)
def multiple_wires(params, inputs):
    # for layer in range(num_layers):
    # data encoding
    qml.AmplitudeEmbedding(features=inputs, wires=range(num_wires), normalize=True)

    for i in range(num_wires):
        qml.RY(params[3 * i], wires=i)
        qml.RZ(params[3 * i + 1], wires=i)

    # entangle the output wires with all other ones
    output_wires = range(num_outputs)
    for wire in output_wires:
        for i in range(num_wires):
            # qml.CNOT(wires=[i + num_outputs, wire])
            if not i == wire:
                qml.CRY(params[3 * i + 2], wires=[i, wire])

    # for i in range(num_wires):
    #     qml.RY(params[i] * inputs[i], wires=i)

    # measure the output wires
    outputs = []
    for wire in output_wires:
        outputs.append(qml.expval(qml.PauliZ(wires=wire)))

    # TODO: Add multiple layers ?

    return outputs

@qml.qnode(shot_dev)
def multiple_shots_outputs(params, inputs):
    qml.AmplitudeEmbedding(features=inputs, wires=range(num_wires), normalize=True)

    for i in range(num_wires):
        qml.RY(params[3 * i], wires=i)
        qml.RZ(params[3 * i + 1], wires=i)

    # entangle the output wires with all other ones
    output_wires = range(num_outputs)
    for wire in output_wires:
        for i in range(num_wires):
            if not i == wire:
                qml.CRY(params[3 * i + 2], wires=[i, wire])

    # measure the output wires
    outputs = [qml.expval(qml.PauliZ(wires=wire)) for wire in output_wires]

    return outputs

@qml.qnode(shot_dev)
def multiple_shots_probs(params, inputs):
    qml.AmplitudeEmbedding(features=inputs, wires=range(num_wires), normalize=True)
    for i in range(num_wires):
        qml.RY(params[3 * i], wires=i)
        qml.RZ(params[3 * i + 1], wires=i)

    # entangle the output wires with all other ones
    output_wires = range(num_outputs)
    for wire in output_wires:
        for i in range(num_wires):
            if not i == wire:
                qml.CRY(params[3 * i + 2], wires=[i, wire])

    # calculate the probabilities of the quantum states
    probabilities = qml.probs(wires=range(num_wires))

    return probabilities


@qml.qnode(shot_dev)
def multiple_shots_outputs(params, inputs):
    qml.AmplitudeEmbedding(features=inputs, wires=range(num_wires), normalize=True)
    for i in range(num_wires):
        qml.RY(params[3 * i], wires=i)
        qml.RZ(params[3 * i + 1], wires=i)

    # entangle the output wires with all other ones
    output_wires = range(num_outputs)
    for wire in output_wires:
        for i in range(num_wires):
            if not i == wire:
                qml.CRY(params[3 * i + 2], wires=[i, wire])

    # measure the output wires
    outputs = [qml.expval(qml.PauliZ(wires=wire)) for wire in output_wires]

    return outputs
#https://mathoverflow.net/questions/244293/generalisations-of-the-kullback-leibler-divergence-for-more-than-two-distributio



# i = [0,1,2,3,4,5,6,7,8,9]
# p = np.random.rand(10)
# print(qml.draw(multiple_wires(p, i)))
# print(multiple_wires(p, i))
