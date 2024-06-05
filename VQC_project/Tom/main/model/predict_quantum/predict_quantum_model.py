from keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
import pennylane as qml
import tensorflow as tf


class PQModel:
    def __init__(self, config):
        self.config = config
        quantum_circuit, weight_shapes = create_pq_quantum_circuit(config)
        self.model = create_pq_model(quantum_circuit, weight_shapes, config)

    def train(self, x_train, y_train):
        history = self.model.fit(x_train, y_train, epochs=self.config['epochs'], batch_size=self.config['batch_size'])
        return history

    def evaluate(self, x_test, y_test):
        loss = self.model.evaluate(x_test, y_test)
        return loss

    def predict(self, x):
        return self.model.predict(x)


def create_pq_model(quantum_circuit, weight_shapes, config):
    inputs = Input(shape=(config['time_steps'], 1))

    # VQC layer, reshape the output to remove the complex part
    quantum_layer = qml.qnn.KerasLayer(quantum_circuit, weight_shapes, output_dim=1)(inputs)
    quantum_layer = tf.reshape(quantum_layer, (-1, 1))
    output = Dense(config['future_steps'], activation='linear')(quantum_layer)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=config['learning_rate']), loss=config['loss_function'])
    return model


def create_pq_quantum_circuit(config):
    dev = qml.device("default.qubit", wires=config['time_steps'])

    @qml.qnode(dev, interface='tf')
    def quantum_circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(config['time_steps']))
        qml.StronglyEntanglingLayers(weights, wires=range(config['time_steps']))
        return qml.expval(qml.PauliZ(0))

    weight_shapes = {"weights": (config['n_layers'], config['time_steps'], 3)}
    return quantum_circuit, weight_shapes
