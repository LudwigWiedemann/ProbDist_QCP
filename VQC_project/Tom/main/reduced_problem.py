from functools import partial

from keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
from pennylane import numpy as np
import tensorflow as tf
import pennylane as qml


# Hardcoded Function to create Timeseries dataset, only needed for execution shouldn't be relevant
def generate_time_series_data():
    time_steps = 32
    num_samples = 256
    future_steps = 5
    noise_level = 0.1
    n_steps = 200

    # Ensure the length of x_data is greater than time_steps
    n_steps = max(time_steps + future_steps, n_steps)  # Ensure a minimum length
    x_data = np.linspace(-4 * np.pi, 12 * np.pi,
                         n_steps)
    y_data = np.sin(x_data)
    noisy_y_data = y_data + np.random.normal(0, noise_level, size=y_data.shape)

    input_train = []
    output_train = []

    input_noisy_test = []
    output_test = []
    input_real_test = []

    if time_steps + future_steps >= len(x_data):
        raise ValueError("time_steps + future_steps must be less than the length of x_data")

    for _ in range(num_samples):
        start_idx = np.random.randint(0, len(x_data) - time_steps - future_steps)
        input_sample = y_data[start_idx:start_idx + time_steps]
        input_noisy_sample = noisy_y_data[start_idx:start_idx + time_steps]
        output_sample = y_data[start_idx + time_steps:start_idx + time_steps + future_steps]

        if np.random.rand() < 0.6:
            input_train.append(input_noisy_sample)
            output_train.append(output_sample)
        else:
            input_noisy_test.append(input_noisy_sample)
            input_real_test.append(input_sample)
            output_test.append(output_sample)

    # Prepare data for foresight
    future_start_idx = len(x_data) - time_steps
    input_foresight = y_data[future_start_idx:future_start_idx + time_steps]
    input_noisy_foresight = input_foresight + np.random.normal(0, noise_level, input_foresight.shape)
    step_size = (-4 * np.pi - 12 * np.pi) / (n_steps - 1)

    input_train = np.array(input_train).reshape(-1, time_steps, 1)
    output_train = np.array(output_train).reshape(-1, future_steps)
    input_real_test = np.array(input_real_test).reshape(-1, time_steps, 1)
    input_noisy_test = np.array(input_noisy_test).reshape(-1, time_steps, 1)
    output_test = np.array(output_test).reshape(-1, future_steps)
    input_foresight = np.array(input_foresight).reshape(1, time_steps, 1)
    input_noisy_foresight = np.array(input_noisy_foresight).reshape(1, time_steps, 1)

    dataset = {'input_train': input_train, 'output_train': output_train, 'input_noisy_test': input_noisy_test,
               'input_test': input_real_test, 'output_test': output_test,
               'input_forecast': input_foresight, 'input_noisy_forecast': input_noisy_foresight, 'step_size': step_size}

    return dataset


# The basic hardcoded wrapper, self-explanatory has a batch_size of 64
class TF_Model:
    def __init__(self, model):
        self.model = model()

    def train(self, dataset):
        x_train = dataset['input_train']
        y_train = dataset['output_train']
        self.model.fit(x_train, y_train, epochs=25, batch_size=32)

    def evaluate(self, dataset):
        x_test = dataset['input_test']
        y_test = dataset['output_test']

        loss = self.model.evaluate(x_test, y_test)
        return loss


#  Tensorflow Model with a basic circuit, needs a different shape before and after the qml layer
def tf_model_simple_circuit():
    circuit, weights = get_circuit_and_weights()

    # Hardcoded to 32 because of the train dataset structure has 32 values per sample
    inputs = Input(shape=(32, 1))

    # Layer to transform shape, hardcoded to 1 because example circuit has 1 wire
    transform_layer = Dense(1, activation='linear')(inputs)

    # This layer is the interface for the pennylane circuit
    quantum_layer = qml.qnn.KerasLayer(circuit, weights, output_dim=1)(transform_layer)

    # Layer to transform shape, hardcoded to 7 because of the train dataset structure expects 7 values per sample
    outputs = Dense(5, activation='linear')(quantum_layer)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.004), loss='mse')
    return model


# Function to give tensorflow the circuit for the qml layer
def get_circuit_and_weights():
    training_device = qml.device("default.qubit", wires=1)

    # Placeholder for circuits of various complexity
    #@partial(qml.batch_input, argnum=[1, 2, 3])
    @qml.qnode(training_device, interface='tf')
    def example_circuit(inputs, weights_0, weights_1, weights_2):
        # Everything hardcoded for placeholder circuit
        qml.RY(inputs[0], 0)
        qml.RX(weights_0, 0)
        qml.RY(weights_1, 0)
        qml.RZ(weights_2, 0)
        return qml.expval(qml.PauliZ(wires=0))

    weight_shapes = {"weights_0": 1, "weights_1": 1, "weights_2": 1}
    return example_circuit, weight_shapes


# Tensorflow Model with a complex circuit, doesn't need a different shape before and after the qml layer
def tf_model_complex_circuit():
    circuit, weights = get_complex_circuit_and_weights()
    inputs = Input(shape=(32,1))
    reshaped_inputs = tf.keras.layers.Reshape((32,))(inputs)
    quantum_layer = qml.qnn.KerasLayer(circuit, weight_shapes=weights, output_dim=1)(reshaped_inputs)
    model = Model(inputs=inputs, outputs=quantum_layer)
    model.compile(optimizer=Adam(learning_rate=0.004), loss='mse')
    return model


# Function to give the tensorflow model the circuit for the qml layer
def get_complex_circuit_and_weights():
    training_device = qml.device("default.qubit", wires=5)

    @partial(qml.batch_input, argnum=0)
    @qml.qnode(training_device, interface='tf')
    def example_complex_circuit(inputs, weights_RY, weights_RX, weights_RZ):
        qml.AmplitudeEmbedding(features=inputs, wires=range(5), normalize=True)
        qml.broadcast(qml.RY, wires=range(5), pattern="single", parameters=weights_RY)
        qml.broadcast(qml.RX, wires=range(5), pattern="single", parameters=weights_RX)
        qml.broadcast(qml.RZ, wires=range(5), pattern="single", parameters=weights_RZ)
        qml.broadcast(qml.CNOT, wires=range(5), pattern="all_to_all")
        return [qml.expval(qml.PauliZ(i)) for i in range(5)]

    weight_shapes = {"weights_RY": 5,"weights_RX": 5,"weights_RZ": 5}
    return example_complex_circuit, weight_shapes


# Function to evaluate model/circuit on given dataset
def fit_and_evaluate_model(dataset, model_with_circuit):
    model = TF_Model(model_with_circuit)
    # Fit the model
    model.train(dataset)
    # Evaluate the model
    loss = model.evaluate(dataset)
    print(f"Loss: {loss}")


if __name__ == "__main__":
    # Create Dataset for fitting and eva
    dataset = generate_time_series_data()
    # Fit and eva model with complex circuit
    fit_and_evaluate_model(dataset, tf_model_complex_circuit)
    # Fit and eva model with simple circuit
    fit_and_evaluate_model(dataset, tf_model_simple_circuit)
