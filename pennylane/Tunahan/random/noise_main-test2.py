import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import numpy as numpy

# TODO: Create a distribution using shots trying to approximate sin(x) functions with noise
# This is bullshit doesnt work try again later

# Define global variables
num_qubits = 1
num_layers = 9
num_params_per_layer = 1
total_num_params = num_layers * num_params_per_layer
num_training_points = 50
training_inputs = np.linspace(0, 10, num_training_points)
training_iterations = 200
training_data = None
opt = qml.GradientDescentOptimizer(0.01)
noise = None
dev = None
circuit = None

# Flags
is_statistical = None


def init_params(stat):
    global num_qubits, num_layers, num_params_per_layer, total_num_params, num_training_points, training_inputs
    global training_iterations, opt, is_statistical, noise, dev, training_data
    # global noise, dev, is_statistical, training_data

    if stat:
        is_statistical = True
        noise = numpy.random.normal(0, 0.1, len(training_inputs))
        dev = qml.device("default.qubit", wires=num_qubits, shots=1000)
    else:
        is_statistical = False
        dev = qml.device("default.qubit", wires=num_qubits)

    training_data = [(x, f(x)) for x in training_inputs]

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

    if is_statistical:
        return qml.sample(qml.PauliZ(wires=0))
    else:
        return qml.expval(qml.PauliZ(wires=0))

def f(x):
    if is_statistical:
        return np.sin(x) + noise
    else:
        # return np.sin(x) * np.cos(2*x)/2*np.sin(x)
        return np.sin(x)
        # return np.sin(x) + 0.5*np.cos(2*x) + 0.25 * np.sin(3*x)


def guess_starting_params_non_stat():
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
def guess_starting_params_stat():
    return guess_starting_params_non_stat()

# Define some functions to use as cost function
def i_have_no_idea(prediction, target):
    return ((prediction - target) ** 2) / 2
def mean_squared_error(prediction, target):
    return np.mean((prediction - target) ** 2)
def cost(params, x, target):
    predicted_output = circuit(params, x)
    return i_have_no_idea(predicted_output, target)


def train_circuit(training_params, num_iterations):
    print("Training the circuit...")
    mean_error_before = None
    for iteration in range(num_iterations):
        errors = []

        for training_x, training_y in training_data:
            training_params = opt.step(cost, training_params, x=training_x, target=training_y)
            predicted_output = circuit(training_params, training_x)
            error = np.abs(predicted_output - training_y)
            errors.append(error)

        if iteration % 10 == 0:
            mean_error = np.mean(errors)
            if mean_error_before is not None:
                print(f"Iteration {iteration}:    Mean Error: {mean_error:.20f},  "
                      f"Error Difference (negative is good): {mean_error - mean_error_before:.20f}")
            else:
                print(f"Iteration {iteration}: Mean Error: {mean_error:.20f}")
            mean_error_before = mean_error
    return training_params


def evaluate_circuit_non_stat(final_params, x_min=-5 * np.pi, x_max=5 * np.pi, n_points=10000):
    print("Evaluating the trained circuit...")
    print("Circuit uses these params:")
    print(final_params)
    x_values = np.linspace(x_min, x_max, n_points)  # Define range for plotting
    x_actual = numpy.linspace(x_min, x_max, n_points)  # Interesting to test the impact of point difference
    actual_ouput = f(x_actual)
    #predicted_outputs = [circuit(final_params, x) for x in x_values]
    predicted_outputs = circuit(final_params, x_values)  # vectorized way faster than for loop
    #plt.ylim(-2, 2)
    plt.grid(True)
    plt.plot(x_actual, actual_ouput, label="Actual f(x)")
    plt.plot(x_values, predicted_outputs, label="Predicted f(x)")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Actual vs. Predicted function f(x)")
    plt.show()
def evaluate_circuit_stat(final_params, x_min=-5 * np.pi, x_max=5 * np.pi, n_points=10000):
    print("Evaluating the trained circuit...")
    print("Circuit uses these params:")
    print(final_params)
    x_values = np.linspace(x_min, x_max, n_points)  # Define range for plotting
    x_actual = numpy.linspace(x_min, x_max, n_points)  # Interesting to test the impact of point difference
    actual_ouput = f(x_actual)
    predicted_outputs = [circuit(final_params, x) for x in x_values]
    # Count the frequency of each output state
    unique, counts = np.unique(predicted_outputs, return_counts=True)
    # Normalize the counts to get probabilities
    probabilities = counts / np.sum(counts)
    plt.grid(True)
    plt.bar(unique, probabilities, label="Predicted distribution")
    plt.legend()
    plt.xlabel("Output state")
    plt.ylabel("Probability")
    plt.title("Predicted distribution")
    plt.show()


def save_params(params, filename, path='Saved Circuit Models'):
    with open(f'{path}/{filename}', 'w') as f:
        f.write(numpy.array2string(params, separator=','))
def load_params(filename):
    with open(f'Saved Circuit Models/{filename}', 'r') as f:
        params_str = f.read()
    params_str = params_str.strip('[]\n')
    params = numpy.fromstring(params_str, sep=',')
    return params

if __name__ == '__main__':
    while True:
        is_statistical = input("Is your data statistical? (Enter 'y' for yes and 'n' for no): ").lower() in ["yes", "y"]
        user_input_trained_model = input("Do you want to load a trained model? "
                                         "(Enter 'y' for yes and 'n' for no): ").lower() in ["yes", "y"]
        init_params(is_statistical)
        if user_input_trained_model:
            user_input_filename = input("Enter the filename: ")
            trained_params = load_params(user_input_filename)
            while True:
                if is_statistical:
                    evaluate_circuit_stat(trained_params)
                else:
                    if input("Change the plot settings? (Enter 'y' or 'yes' if so): ").lower() == "yes" or "y":
                        x_min = float(input("Enter the minimum x value: "))
                        x_max = float(input("Enter the maximum x value: "))
                        n_points = int(input("Enter the number of points: "))
                        evaluate_circuit_non_stat(trained_params, x_min, x_max, n_points)
                    else:
                        evaluate_circuit_non_stat(trained_params)

                    plot_again = input("Do you want to plot with the same circuit again? "
                                       "(Enter 'y' for yes and 'n' for no): ")
                    if plot_again.lower() == 'n' or plot_again.lower() == 'no':
                        break
        else:
            user_input_mode = input("Please select mode:\n"
                                    "v || variable loop -> Variable number of iterations\n"
                                    "s || single -> Single predefined run (200 iterations)\n"
                                    "dl || discrete loop -> Discrete loops (200 iterations per loop)\n")
            if user_input_mode == "s" or "single":
                # Call the function for a single run
                if is_statistical:
                    starting_params = guess_starting_params_stat()
                    trained_params = train_circuit(starting_params, training_iterations)
                    evaluate_circuit_stat(trained_params)
                else:
                    starting_params = guess_starting_params_non_stat()
                    trained_params = train_circuit(starting_params, training_iterations)
                    evaluate_circuit_non_stat(trained_params)
            elif user_input_mode == "dl" or "discrete loop":
                # Call the function for a loop
                user_input_loop = input("How many loops do you want to run? ")
                user_input_save = input("Do you want to save the trained parameters? "
                                        "(Enter 'y' for yes and 'n' for no): ").lower()
                user_input_filename = input("Enter the filename: ")
                if is_statistical:
                    for i in range(int(user_input_loop)):
                        starting_params = guess_starting_params_stat()
                        trained_params = train_circuit(starting_params, training_iterations)
                        evaluate_circuit_stat(trained_params)
                        if user_input_save == "yes" or "y":
                            user_input_filename_counted = user_input_filename + "_loop_" + str(i)
                            save_params(trained_params, user_input_filename_counted,
                                        path='Saved Circuit Models/Discrete Loops')
                else:
                    for i in range(int(user_input_loop)):
                        starting_params = guess_starting_params_non_stat()
                        trained_params = train_circuit(starting_params, training_iterations)
                        evaluate_circuit_non_stat(trained_params)
                        if user_input_save == "yes" or "y":
                            user_input_filename_counted = user_input_filename + "_loop_" + str(i)
                            save_params(trained_params, user_input_filename_counted,
                                        path='Saved Circuit Models/Discrete Loops')
            elif user_input_mode == "v" or "variable loop":
                # Call the function for a loop
                training_iterations = int(input("How many iterations do you want to run? "))
                user_input_save = input("Do you want to save the trained parameters? "
                                        "(Enter 'y' for yes and 'n' for no): ")
                if is_statistical:
                    if user_input_save == "yes" or "y":
                        user_input_filename = input("Enter the filename: ")
                        starting_params = guess_starting_params_stat()
                        trained_params = train_circuit(starting_params, training_iterations)
                        evaluate_circuit_stat(trained_params)
                        save_params(trained_params, user_input_filename)
                    else:
                        starting_params = guess_starting_params_stat()
                        trained_params = train_circuit(starting_params, training_iterations)
                        evaluate_circuit_stat(trained_params)
                else:
                    if user_input_save == "yes" or "y":
                        user_input_filename = input("Enter the filename: ")
                        starting_params = guess_starting_params_non_stat()
                        trained_params = train_circuit(starting_params, training_iterations)
                        evaluate_circuit_non_stat(trained_params)
                        save_params(trained_params, user_input_filename)
                    else:
                        starting_params = guess_starting_params_non_stat()
                        trained_params = train_circuit(starting_params, training_iterations)
                        evaluate_circuit_non_stat(trained_params)
            else:
                print("Invalid input. Please try again")
