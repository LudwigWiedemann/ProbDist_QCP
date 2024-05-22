import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt

# Define problem parameters
num_qubits = 1

num_layers = 9
num_params_per_layer = 1
total_num_params = num_layers * num_params_per_layer

num_training_points = 10  # Increase the number of training points
training_inputs = np.linspace(0, 10, num_training_points)  # Use np.linspace for even distribution
training_iterations = 20
reruns = 5  #how many times should this be evaluated
prediction = 100  #how much should be predicted

dev = qml.device("default.qubit", wires=num_qubits)
opt = qml.GradientDescentOptimizer(0.001)
trained_params_list = []


def f(x):
    return np.sin(x)  #  return np.sin(x) + 0.5*np.cos(2*x) + 0.25 * np.sin(3*x)


def guess_starting_params():
    print("guessing best starting parameters ... ")
    num_attempts = 3
    attempts = [[], [], []]
    errors = [99, 99, 99]
    for i in range(num_attempts - 1):
        attempts[i] = np.random.rand(total_num_params)
        x0 = 0
        x1 = np.pi
        cost_x0 = int(cost(attempts[i], x0, f(x0)))
        cost_x1 = int(cost(attempts[i], x1, f(x1)))
        mean_error = np.mean([cost_x0, cost_x1])
        errors[i] = mean_error

    best_attempt = 0
    for i in range(len(errors)):
        if errors[i] < errors[best_attempt]:
            best_attempt = i
    print("Best params: " + str(attempts[best_attempt]))
    return attempts[best_attempt]


@qml.qnode(dev)
def circuit(params, x):
    # for i in range(num_layers):
    #     qml.RY(params[i] * x, wires=0)
    #     qml.RY(params[i + 1] + x, wires=0)
    qml.RY(params[0] * x, wires=0)
    qml.RY(params[1] * x, wires=0)
    qml.RY(params[2], wires=0)
    qml.RY(params[3], wires=0)
    qml.RY(params[4] * x, wires=0)
    qml.RY(params[5], wires=0)
    qml.RY(params[6], wires=0)
    qml.RY(params[7], wires=0)
    qml.RY(params[8], wires=0)
    # qml.RY(params[2] / x, wires=0) geht nicht weil kein int

    return qml.expval(qml.PauliZ(wires=0))


# Define the mean squared error cost function


def cost(params, x, target):
    predicted_output = circuit(params, x)
    return ((predicted_output - target) ** 2) / 2


def train_circuit(training_params, num_iterations, prediction_num):
    print("Training the circuit "+str(prediction_num)+"...")
    runinfo = str(prediction_num)+"/"+str(reruns)
    for iteration in range(num_iterations):
        for training_x, training_y in training_data:
            training_params = opt.step(cost, training_params, x=training_x, target=training_y)
        if iteration % 10 == 0:
            print(f"Iteration {iteration}:")
            for training_x, training_y in training_data:
                predicted_output = circuit(training_params, training_x)
                error = np.abs(predicted_output - training_y)
                print(
                    f"Run: {runinfo} Input: {training_x}, Expected: {training_y:.4f}, Predicted: {predicted_output:.4f}, Error: {error:.4f}")
    return training_params


def evaluate_circuit(params_list: list):
    print("Evaluating the trained circuit...")
    x_values = np.linspace(-3 * np.pi, 6 * np.pi+prediction, 100)  # Define range for plotting
    actual_ouput = f(x_values)
    plt.ylim(-2, 2)
    plt.grid(True)
    plt.plot(x_values, actual_ouput, label="Actual f(x)")
    for i in range(0 , len(params_list)):
        final_params = params_list[i]
        print(final_params)
        predicted_outputs = [circuit(final_params, x) for x in x_values]
        plt.plot(x_values, predicted_outputs, label="predicted_output"+str(i))
    #plt.plot(x_values, avarage_output, label="Avarage")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("Sin(x)")
    plt.title("Actual vs. Predicted Sin")
    plt.show()


training_data = [(x, f(x)) for x in training_inputs]

for j in range(reruns):
    starting_params = guess_starting_params()
    trained_params = train_circuit(starting_params, training_iterations, j+1)
    trained_params_list.append(trained_params)

#average_params = circuit(np.array(trained_params_list).mean(axis=0), x=training_inputs)
#print(average_params)
#print(trained_params_list)
evaluate_circuit(trained_params_list)
