import pennylane as qml
from pennylane import numpy as np
import matplotlib.pyplot as plt
import numpy as numpy
import os


class QuantumMlAlgorithm:

    @staticmethod
    def f(x):
        # return np.sin(x) + noise
        # return np.sin(x) * np.cos(2*x)/2*np.sin(x)
        return np.sin(x)
        # return np.sin(x) + 0.5*np.cos(2*x) + 0.25 * np.sin(3*x)

    def __init__(self):
        # Define problem parameters
        numpy.random.seed(200)
        self.num_qubits = 1
        self.num_layers = 9
        self.num_params_per_layer = 1
        self.total_num_params = self.num_layers * self.num_params_per_layer

        self.num_training_points = 50  # Increase the number of training points
        self.training_inputs = np.linspace(0, 10, self.num_training_points)  # Use np.linspace for even distribution
        self.training_iterations = 200
        self.noise = 0.1 * np.random.randn(len(self.training_inputs))

        self.dev = qml.device("default.qubit", wires=self.num_qubits)
        self.opt = qml.GradientDescentOptimizer(0.001)
        self.training_data = [(x, self.f(x)) for x in self.training_inputs]

    def guess_starting_params(self):
        print("guessing best starting parameters ... ")
        num_attempts = 3
        attempts = [[], [], []]
        errors = [99, 99, 99]
        for i in range(num_attempts - 1):
            attempts[i] = np.random.rand(self.total_num_params)
            x0 = 0
            x1 = np.pi
            cost_x0 = int(self.cost(attempts[i], x0, self.f(x0)))
            cost_x1 = int(self.cost(attempts[i], x1, self.f(x1)))
            mean_error = np.mean([cost_x0, cost_x1])
            errors[i] = mean_error

        best_attempt = 0
        for i in range(len(errors)):
            if errors[i] < errors[best_attempt]:
                best_attempt = i
        print("Best params: " + str(attempts[best_attempt]))
        return attempts[best_attempt]

    def circuit(self, params, x):
        @qml.qnode(self.dev)
        def _circuit(params, x):
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
        return _circuit(params, x)

    # Define some functions to use as cost function
    @staticmethod
    def i_have_no_idea(prediction, target):
        return ((prediction - target) ** 2) / 2

    @staticmethod
    def mean_squared_error(prediction, target):
        return np.mean((prediction - target) ** 2)

    def cost(self, params, x, target):
        predicted_output = self.circuit(params, x)
        return self.i_have_no_idea(predicted_output, target)

    def train_circuit(self, training_params, num_iterations):
        print("Training the circuit...")
        mean_error_before = None
        for iteration in range(num_iterations):
            errors = []

            for training_x, training_y in self.training_data:
                training_params = self.opt.step(self.cost, training_params, x=training_x, target=training_y)
                predicted_output = self.circuit(training_params, training_x)
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

    def evaluate_circuit(self, final_params, x_min=-5 * np.pi, x_max=5 * np.pi, n_points=10000):
        print("Evaluating the trained circuit...")
        print("Circuit uses these params:")
        print(final_params)
        x_values = np.linspace(x_min, x_max, n_points)  # Define range for plotting
        x_actual = numpy.linspace(x_min, x_max, n_points)  # Interesting to test the impact of point difference
        actual_ouput = self.f(x_actual)
        #predicted_outputs = [circuit(final_params, x) for x in x_values]
        predicted_outputs = self.circuit(final_params, x_values)  # vectorized way faster than for loop
        #plt.ylim(-2, 2)
        plt.grid(True)
        plt.plot(x_actual, actual_ouput, label="Actual f(x)")
        plt.plot(x_values, predicted_outputs, label="Predicted f(x)")
        plt.legend()
        plt.xlabel("x")
        plt.ylabel("f(x)")
        plt.title("Actual vs. Predicted function f(x)")
        plt.show()

    def save_params(self, params, filename, path='Saved Circuit Models'):
        with open(f'{path}/{filename}', 'w') as f:
            f.write(numpy.array2string(params, separator=','))

    def load_params(self, filename):
        with open(f'Saved Circuit Models/{filename}', 'r') as f:
            params_str = f.read()
        params_str = params_str.strip('[]\n')
        params = numpy.fromstring(params_str, sep=',')
        return params


if __name__ == '__main__':
    alg = QuantumMlAlgorithm()
    while True:
        user_input_start = input("Do you want to load a trained model? "
                                 "(Enter 'y' for yes and 'n' for no): ").lower() in ["yes", "y"]
        if user_input_start:
            try:
                filenames = os.listdir('Saved Circuit Models')
                for i, filename in enumerate(filenames):
                    print(f"{i}: {filename}")
                user_input_filename = filenames[int(input("Enter the number of the file: "))]
                trained_params = alg.load_params(user_input_filename)
            except Exception as e:
                print(f"An error occurred: {e}")
                continue
            while True:
                if input("Change the plot settings? (Enter 'y' or 'yes' if so): ").lower() in ["yes", "y"]:
                    x_min = float(input("Enter the minimum x value: "))
                    x_max = float(input("Enter the maximum x value: "))
                    n_points = int(input("Enter the number of points: "))
                    alg.evaluate_circuit(trained_params, x_min, x_max, n_points)
                else:
                    alg.evaluate_circuit(trained_params)

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
                starting_params = alg.guess_starting_params()
                trained_params = alg.train_circuit(starting_params, training_iterations)
                alg.evaluate_circuit(trained_params)

            elif user_input.lower() == 'dl' or user_input.lower() == 'discrete loop':
                # Call the function for a loop
                user_input_loop = input("How many loops do you want to run? ")
                user_input_save = input("Do you want to save the trained parameters? "
                                        "(Enter 'y' for yes and 'n' for no): ")
                user_input_filename = input("Enter the filename: ")
                for i in range(int(user_input_loop)):
                    starting_params = alg.guess_starting_params()
                    trained_params = alg.train_circuit(starting_params, training_iterations)
                    alg.evaluate_circuit(trained_params)

                    if user_input_save.lower() == 'y' or user_input_save.lower() == 'yes':
                        user_input_filename_counted = user_input_filename + "_loop_" + str(i)
                        alg.save_params(trained_params, user_input_filename_counted,
                                        path='Saved Circuit Models/Discrete Loops')

            elif user_input.lower() == 'v' or user_input.lower() == 'variable loop':
                # Call the function for a loop
                training_iterations = int(input("How many loops do you want to run? "))
                user_input_save = input("Do you want to save the trained parameters? "
                                        "(Enter 'y' for yes and 'n' for no): ")

                if user_input_save.lower() == 'y' or user_input_save.lower() == 'yes':
                    user_input_filename = input("Enter the filename: ")
                    starting_params = alg.guess_starting_params()
                    trained_params = alg.train_circuit(starting_params, training_iterations)
                    alg.evaluate_circuit(trained_params)
                    alg.save_params(trained_params, user_input_filename)

                else:
                    starting_params = alg.guess_starting_params()
                    trained_params = alg.train_circuit(starting_params, training_iterations)
                    alg.evaluate_circuit(trained_params)

            else:
                print("Invalid input. Please enter 'l' for loop and 's' for single run.")
