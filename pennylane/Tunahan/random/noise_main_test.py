import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import numpy as numpy

# TODO: Create a distribution using many circuits trying to approximate sin(x) functions with noise

# Define problem parameters
num_qubits = 1
num_layers = 9
num_params_per_layer = 1
total_num_params = num_layers * num_params_per_layer

num_training_points = 50  # Increase the number of training points
training_inputs = np.linspace(0, 10, num_training_points)  # Use np.linspace for even distribution
training_iterations = 200
noise = 0.1*numpy.random.randn(len(training_inputs))

dev = qml.device("default.qubit", wires=num_qubits)
opt = qml.GradientDescentOptimizer(0.001)


def f(x):
    return np.sin(x) + noise
    # return np.sin(x) * np.cos(2*x)/2*np.sin(x)
    # return np.sin(x)
    # return np.sin(x) + 0.5*np.cos(2*x) + 0.25 * np.sin(3*x)

training_data = [(x, f(x)) for x in training_inputs]

def guess_starting_params():
    print("guessing best starting parameters ... ")
    num_attempts = 3
    attempts = [[], [], []]
    errors = [99, 99, 99]
    num_points = 10
    x_values = np.linspace(0, np.pi, num_points)  # Use more points for the initial guess

    for i in range(num_attempts - 1):
        attempts[i] = np.random.rand(total_num_params)
        costs = [cost(attempts[i], x, f(x)) for x in x_values]  # Calculate the cost for each point
        mean_error = np.mean(costs)  # Use the mean cost as the error
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


def evaluate_circuit(final_params, x_min=-5 * np.pi, x_max=5 * np.pi, n_points=10000):
    print("Evaluating the trained circuit...")
    print("Circuit uses these params:")
    print(final_params)
    x_values = np.linspace(x_min, x_max, n_points)  # Define range for plotting
    x_actual = numpy.linspace(x_min, x_max, n_points)  # Interesting to test the impact of point difference
    actual_ouput = f(x_actual)
    #predicted_outputs = [circuit(final_params, x) for x in x_values]
    predicted_outputs = circuit(final_params, x_values) # vectorized way faster than for loop
    #plt.ylim(-2, 2)
    plt.grid(True)
    plt.plot(x_actual, actual_ouput, label="Actual f(x)")
    plt.plot(x_values, predicted_outputs, label="Predicted f(x)")
    plt.legend()
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.title("Actual vs. Predicted function f(x)")
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
        user_input_start = input("Do you want to load a trained model? (Enter 'y' for yes and 'n' for no): ")
        if user_input_start.lower() == 'y' or user_input_start.lower() == 'yes':
            user_input_filename = input("Enter the filename: ")
            trained_params = load_params(user_input_filename)
            while True:
                if input("Change the plot settings? (Enter 'y' or 'yes' if so): ").lower() == "yes" or "y":
                    x_min = float(input("Enter the minimum x value: "))
                    x_max = float(input("Enter the maximum x value: "))
                    n_points = int(input("Enter the number of points: "))
                    evaluate_circuit(trained_params, x_min, x_max, n_points)
                else:
                    evaluate_circuit(trained_params)

                plot_again = input("Do you want to plot with the same circuit again? "
                                   "(Enter 'y' for yes and 'n' for no): ")
                if plot_again.lower() == 'n' or plot_again.lower() == 'no':
                    break
        else:
            user_input = input("Please select mode:\n"
                               "v || variable loop -> Variable number of iterations\n"
                               "s || single -> Single predefined run (200 iterations)\n"
                               "dl || discrete loop -> Discrete loops (200 iterations per loop)\n")

            if user_input.lower() == 's' or user_input.lower() == 'single':
                # Call the function for a single run
                starting_params = guess_starting_params()
                trained_params = train_circuit(starting_params, training_iterations)
                evaluate_circuit(trained_params)

            elif user_input.lower() == 'dl' or user_input.lower() == 'discrete loop':
                # Call the function for a loop
                user_input_loop = input("How many loops do you want to run? ")
                user_input_save = input("Do you want to save the trained parameters? "
                                        "(Enter 'y' for yes and 'n' for no): ")
                user_input_filename = input("Enter the filename: ")
                for i in range(int(user_input_loop)):
                    starting_params = guess_starting_params()
                    trained_params = train_circuit(starting_params, training_iterations)
                    evaluate_circuit(trained_params)

                    if user_input_save.lower() == 'y' or user_input_save.lower() == 'yes':
                        user_input_filename_counted = user_input_filename + "_loop_" + str(i)
                        save_params(trained_params, user_input_filename_counted,
                                    path='Saved Circuit Models/Discrete Loops')

            elif user_input.lower() == 'v' or user_input.lower() == 'variable loop':
                # Call the function for a loop
                training_iterations = int(input("How many loops do you want to run? "))
                user_input_save = input("Do you want to save the trained parameters? "
                                        "(Enter 'y' for yes and 'n' for no): ")

                if user_input_save.lower() == 'y' or user_input_save.lower() == 'yes':
                    user_input_filename = input("Enter the filename: ")
                    starting_params = guess_starting_params()
                    trained_params = train_circuit(starting_params, training_iterations)
                    evaluate_circuit(trained_params)
                    save_params(trained_params, user_input_filename)

                else:
                    starting_params = guess_starting_params()
                    trained_params = train_circuit(starting_params, training_iterations)
                    evaluate_circuit(trained_params)

            else:
                print("Invalid input. Please enter 'l' for loop and 's' for single run.")



