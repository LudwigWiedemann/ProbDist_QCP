import pennylane as qml
import random

num_qubits = 1
num_layers = 9
device = qml.device("default.qubit", wires=num_qubits)


@qml.qnode(device)
def run_circuit(params, x):
    qml.RY(params[0] * x, wires=0)
    qml.RY(params[1] * x, wires=0)
    qml.RY(params[2], wires=0)
    qml.RY(params[3], wires=0)
    qml.RY(params[4] * x, wires=0)
    qml.RY(params[5], wires=0)
    qml.RY(params[6], wires=0)
    qml.RY(params[7], wires=0)
    qml.RY(params[8], wires=0)
    return qml.expval(qml.PauliZ(wires=0))


@qml.qnode(device)
def run_seeded_circuit(params, x, seed):
    for elements in seed:
        layer, cir = elements
        xyz, arithmetic = cir[0], cir[1:]
        code_line = "qml.RY(params[" + str(layer) + "]" + str(arithmetic) + ", wires=0)"
        exec(code_line)
        print(code_line)
    return qml.expval(qml.PauliZ(wires=0))


def read_circuit(seed: [int, str]):
    s: str = ""
    for elements in seed:
        layer, cir = elements
        xyz, arithmetic = cir[0], cir[1:]
        s += "R"+str(xyz)+"(params["+str(layer)+"]"+str(arithmetic)+") "
    print(s)


def randomize_circuit():
    seed = []
    for i in range(num_layers):
        #random trakes for RX
        if random.choice([True, False]):
            seed.append([i, random.choice(['X*1', 'X+x', 'X-x', 'X*x', 'X**x'])])
        #random trakes for RY
        if random.choice([True, False]):
            seed.append([i, random.choice(['Y*1', 'Y+x', 'Y-x', 'Y*x', 'Y**x'])])
        #random trakes for RZ
        if random.choice([True, False]):
            seed.append([i, random.choice(['Z*1', 'Z+x', 'Z-x', 'Z*x', 'Z**x'])])
    return seed

print(randomize_circuit())