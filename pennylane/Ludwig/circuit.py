import pennylane as qml
import simulation as sim
from simulation import full_config as conf

num_outputs = conf['future_steps']
num_wires = sim.num_wires
num_shots = conf['num_shots_for_evaluation']
num_layers = conf['num_layers']

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

    # measure the output wires
    outputs = []
    for wire in output_wires:
        outputs.append(qml.expval(qml.PauliZ(wires=wire)))

    # TODO: Add multiple layers ?

    return outputs

@qml.qnode(shot_dev)
def multiple_shots(params, inputs):
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

    outputs = []
    for wire in output_wires:
        outputs.append(qml.expval(qml.PauliZ(wires=wire)))

    # TODO: Add multiple layers ?

    return outputs

@qml.qnode(dev)
def poc(weights, inputs):
    for layer in range(num_layers):

        qml.AmplitudeEmbedding(features=inputs, wires=range(num_wires), normalize=True)

        qml.RZ(weights[layer + 0], wires=0)
        qml.RX(weights[layer + 1], wires=0)
        qml.RY(weights[layer + 2], wires=0)

        qml.RZ(weights[layer + 3], wires=1)
        qml.RX(weights[layer + 4], wires=1)
        qml.RY(weights[layer + 5], wires=1)

        qml.RZ(weights[layer + 6], wires=2)
        qml.RX(weights[layer + 7], wires=2)
        qml.RY(weights[layer + 8], wires=2)

        for wire in range(num_wires):
            for other_wire in range(num_wires):
                if wire != other_wire:
                    qml.CNOT(wires=[wire, other_wire])
    outputs = []
    for wire in range(num_outputs):
        outputs.append(qml.expval(qml.PauliZ(wires=wire)))
    return outputs


@qml.qnode(shot_dev)
def shot_poc(weights, inputs):
    qml.AmplitudeEmbedding(features=inputs, wires=range(num_wires), normalize=True)
    for layer in range(num_layers):


        qml.RZ(weights[layer + 0], wires=0)
        qml.RY(weights[layer + 1], wires=0)
        qml.RY(weights[layer + 2], wires=0)

        qml.RZ(weights[layer + 3], wires=1)
        qml.RY(weights[layer + 4], wires=1)
        qml.RY(weights[layer + 5], wires=1)

        qml.RZ(weights[layer + 6], wires=2)
        qml.RY(weights[layer + 7], wires=2)
        qml.RY(weights[layer + 8], wires=2)

        # qml.RZ(weights[layer + 9], wires=3)
        # qml.RY(weights[layer + 10], wires=3)
        # qml.RY(weights[layer + 11], wires=3)

        for wire in range(num_wires):
            for other_wire in range(num_wires):
                if wire != other_wire:
                    qml.CNOT(wires=[wire, other_wire])
    outputs = []
    for wire in range(num_outputs):
        outputs.append(qml.expval(qml.PauliZ(wires=wire)))
    return outputs
