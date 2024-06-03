from keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.optimizers import Adam
import pennylane as qml
import tensorflow as tf


class PHModel:
    def __init__(self, config):
        self.config = config
        quantum_circuit, weight_shapes = create_ph_quantum_circuit(config)
        self.model = create_ph_model(quantum_circuit, weight_shapes, config)

    def train(self, x_train, y_train):
        history = self.model.fit(x_train, y_train, epochs=self.config['epochs'], batch_size=self.config['batch_size'])
        return history

    def evaluate(self, x_test, y_test):
        loss = self.model.evaluate(x_test, y_test)
        return loss

    def predict(self, x):
        return self.model.predict(x)


def create_ph_model(quantum_circuit, weight_shapes, config):
    inputs = Input(shape=(config['time_steps'], config['input_dim']))
    lstm1 = LSTM(100, return_sequences=True)(inputs)
    lstm2 = LSTM(100)(lstm1)
    dense1 = Dense(20, activation='relu')(lstm2)
    dense2 = Dense(5, activation='relu')(dense1)

    # VQC layer, reshape the output to remove the complex part
    quantum_layer = qml.qnn.KerasLayer(quantum_circuit, weight_shapes, output_dim=1)(dense2)
    quantum_layer = tf.reshape(quantum_layer, (-1, 1))

    dense3 = Dense(10, activation='relu')(quantum_layer)
    dense4 = Dense(config['future_steps'], activation='linear')(dense3)

    model = Model(inputs=inputs, outputs=dense4)
    model.compile(optimizer=Adam(learning_rate=config['learning_rate']), loss=config['loss_function'])
    return model

def create_ph_model(quantum_circuit, weight_shapes, config):
    inputs = Input(shape=(config['time_steps'], config['input_dim']))
    lstm1 = LSTM(100, return_sequences=True)(inputs)
    lstm2 = LSTM(100)(lstm1)
    dense1 = Dense(20, activation='relu')(lstm2)
    dense2 = Dense(5, activation='relu')(dense1)

    # VQC layer, reshape the output to remove the complex part
    quantum_layer = qml.qnn.KerasLayer(quantum_circuit, weight_shapes, output_dim=1)(dense2)
    quantum_layer = tf.reshape(quantum_layer, (-1, 1))

    dense3 = Dense(10, activation='relu')(quantum_layer)
    dense4 = Dense(config['future_steps'], activation='linear')(dense3)

    model = Model(inputs=inputs, outputs=dense4)
    model.compile(optimizer=Adam(learning_rate=config['learning_rate']), loss=config['loss_function'])
    return model


def create_ph_quantum_circuit(config):
    dev = qml.device("default.qubit", wires=config['n_qubits'])

    @qml.qnode(dev, interface='tf')
    def quantum_circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(config['n_qubits']))
        qml.StronglyEntanglingLayers(weights, wires=range(config['n_qubits']))
        return qml.expval(qml.PauliZ(0))

    weight_shapes = {"weights": (config['n_layers'], config['n_qubits'], 3)}
    return quantum_circuit, weight_shapes
