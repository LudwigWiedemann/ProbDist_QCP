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
def run_random_circuit(params, x, xyz_seed, arithmetic_seed):
    for i in range(num_layers*3):
        if xyz_seed[i] == "1":
            match i % 3:
                case 1:
                    code_line = "qml.RX(params[" + str(i) + "]"
                case 2:
                    code_line = "qml.RY(params[" + str(i) + "]"
                case 0:
                    code_line = "qml.RZ(params[" + str(i) + "]"
                case _:
                    raise Exception("unexpected Value " + xyz_seed[i] + " for xyz_seed at " + i)
                    break

            match arithmetic_seed[i]:
                case "0":
                    code_line += ", wires=0)"
                case "1":
                    code_line += " * x, wires=0)"
                case "2":
                    code_line += " ** x, wires=0)"
                case "3":
                    code_line += " + x, wires=0)"
                case "4":
                    code_line += " - x, wires=0)"
                case _:
                    raise Exception("unexpected Value " + arithmetic_seed[i] + " for arithmetic_seed at " + i)
                    break
            exec(code_line)
    return qml.expval(qml.PauliZ(wires=0))


def read_circuit(params, x, xyz_seed, arithmetic_seed):
    output = ""
    for i in range(num_layers*3):
        if xyz_seed[i] == "1":
            match i % 3:
                case 1:
                    output += "RX(params[" + str(i) + "]"
                case 2:
                    output += "RY(params[" + str(i) + "]"
                case 0:
                    output += "RZ(params[" + str(i) + "]"
                case _:
                    raise Exception("unexpected Value " + xyz_seed[i] + " for xyz_seed at " + i)
                    break

            match arithmetic_seed[i]:
                case "0":
                    output += ")"
                case "1":
                    output += " * x)"
                case "2":
                    output += " ** x)"
                case "3":
                    output += " + x])"
                case "4":
                    output += " - x)"
                case _:
                    raise Exception("unexpected Value " + arithmetic_seed[i] + " for arithmetic_seed at " + i)
                    break
    return output


def randomize_circuit():
    xyz_seed = ""
    arithmetic_seed = ""
    for i in range(num_layers*3):
        ran = random.randrange(True, False)  #defines if X Y and Z is used
        xyz_seed += str(ran)
        ran = random.randint(0, 4)  #defines if *1, *x, **x, +x or -x is used
        arithmetic_seed += str(ran)
    return xyz_seed, arithmetic_seed
