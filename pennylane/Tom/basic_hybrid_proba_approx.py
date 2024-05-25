import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import pennylane as qml
import tensorflow as tf
import matplotlib.pyplot as plt

from pennylane import numpy as np
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Softmax
from tensorflow.keras.optimizers import Adam

from silence_tensorflow import silence_tensorflow
silence_tensorflow()

# Define the target function
def target_function(x):
    return np.sin(x) + 0.5 * np.cos(2 * x) + 0.25 * np.sin(3 * x)

# Generate training data
def generate_training_data(num_points=50, range_start=0, range_end=10):
    x_train = np.linspace(range_start, range_end, num_points)
    y_train = target_function(x_train)
    return x_train, y_train

# Quantum circuit
def create_quantum_circuit(num_qubits, num_layers):
    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev, interface='tf')
    def quantum_circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(num_qubits))
        qml.StronglyEntanglingLayers(weights, wires=range(num_qubits))
        return qml.probs(wires=range(num_qubits))

    weight_shapes = {"weights": (num_layers, num_qubits, 3)}
    return quantum_circuit, weight_shapes

# Define the Keras model
def create_model(quantum_circuit, weight_shapes):
    inputs = Input(shape=(1,))
    dense1 = Dense(10, activation='relu')(inputs)
    dense2 = Dense(10, activation='relu')(dense1)
    dense3 = Dense(1, activation='linear')(dense2)

    quantum_layer = qml.qnn.KerasLayer(quantum_circuit, weight_shapes, output_dim=2)(dense3)  # Adjust output_dim based on qubits

    # Ensure the output from quantum layer is properly shaped
    quantum_layer = tf.reshape(quantum_layer, (-1, 2))

    # Apply softmax to convert to probabilities
    outputs = Softmax()(quantum_layer)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=Adam(learning_rate=0.01), loss='sparse_categorical_crossentropy')
    return model

# Plot results
def plot_results(x_test, y_test, x_train, y_train, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(x_test.numpy(), y_test, label='True Function', color='blue')
    plt.plot(x_test.numpy(), y_pred[:, 1], label='VQC Approximation', color='red')  # Plotting the probability of class 1
    plt.scatter(x_train.numpy(), y_train.numpy(), color='green', label='Training Points')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Quantum Circuit Learning Probability Distribution using PennyLane and TensorFlow')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Define parameters
    num_qubits = 1
    num_layers = 3

    # Generate training data
    x_train, y_train = generate_training_data()
    x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)
    y_train = tf.convert_to_tensor((y_train > 0).astype(int), dtype=tf.int32)  # Convert y_train to binary classes
    x_train = tf.reshape(x_train, (-1, 1))

    # Create quantum circuit
    quantum_circuit, weight_shapes = create_quantum_circuit(num_qubits, num_layers)

    # Create and train the model
    model = create_model(quantum_circuit, weight_shapes)
    model.fit(x_train, y_train, epochs=200, batch_size=10, verbose=1)

    # Predict and evaluate
    x_test = np.linspace(-3 * np.pi, 6 * np.pi, 100)
    y_test = target_function(x_test)
    x_test = tf.reshape(tf.convert_to_tensor(x_test, dtype=tf.float32), (-1, 1))
    y_pred = model.predict(x_test)

    # Plot results
    plot_results(x_test, y_test, x_train, y_train, y_pred)

if __name__ == "__main__":
    main()
