import pennylane as qml
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam


def create_model(quantum_circuit, weight_shapes, config):
    # Classic layers
    inputs = Input(shape=(1,))
    dense1 = Dense(10, activation='relu')(inputs)
    dense2 = Dense(10, activation='relu')(dense1)
    dense3 = Dense(1, activation='linear')(dense2)

    # VQC layer, reshape the output to remove the complex part
    quantum_layer = qml.qnn.KerasLayer(quantum_circuit, weight_shapes, output_dim=1)(dense3)
    quantum_layer = tf.reshape(quantum_layer, (-1, 1))

    # Further layers
    dense4 = Dense(10, activation='relu')(quantum_layer)
    outputs = Dense(1, activation='linear')(dense4)

    model = Model(inputs=inputs, outputs=outputs)
    # Use Adam as optimizer
    model.compile(optimizer=Adam(learning_rate=config['learning_rate']), loss=config['loss_function'])
    return model

def create_quantum_circuit(config):
    dev = qml.device("default.qubit", wires=config['n_qubits'])
    @qml.qnode(dev, interface='tf')
    def quantum_circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(config['n_qubits']))
        qml.StronglyEntanglingLayers(weights, wires=range(config['n_qubits']))
        return qml.expval(qml.PauliZ(0))

    weight_shapes = {"weights": (config['n_layers'], config['n_qubits'], 3)}
    return quantum_circuit, weight_shapes