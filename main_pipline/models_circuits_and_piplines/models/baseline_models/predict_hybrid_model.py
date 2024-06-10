from keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.optimizers import Adam
from main_pipline.input.div.logger import logger
import pennylane as qml
import tensorflow as tf
from tqdm import tqdm


class PHModel:
    def __init__(self, config):
        circuit = None
        self.config = config

        n_qubits = 5
        n_layers = 5

        quantum_circuit, weight_shapes = create_ph_quantum_circuit(n_qubits, n_layers, config)
        self.model = create_ph_model(n_qubits, quantum_circuit, weight_shapes, config)

    def train(self, dataset):
        x_train = dataset['input_train']
        y_train = dataset['output_train']

        epochs = self.config['epochs']
        batch_size = self.config['batch_size']
        steps_per_epoch = len(x_train) // batch_size

        history = {'loss': []}

        for epoch in tqdm(range(epochs), desc="Training Progress"):
            epoch_loss = 0
            for step in range(steps_per_epoch):
                batch_start = step * batch_size
                batch_end = (step + 1) * batch_size
                x_batch = x_train[batch_start:batch_end]
                y_batch = y_train[batch_start:batch_end]

                batch_loss = self.model.train_on_batch(x_batch, y_batch)
                epoch_loss += batch_loss

            history['loss'].append(epoch_loss / steps_per_epoch)
            logger.info(f"\nEpoch {epoch + 1}/{epochs} Loss: {epoch_loss / steps_per_epoch}")
            tqdm.write(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / steps_per_epoch}")
        return history

    def evaluate(self, dataset):
        x_test = dataset['input_test']
        y_test = dataset['output_test']

        pred_y_test_data = self.model.predict(x_test)
        loss = self.model.evaluate(x_test, y_test)
        return pred_y_test_data, loss

    def predict(self, x_test):
        return self.model.predict(x_test)


def create_ph_model(n_qubits, quantum_circuit, weight_shapes, config):
    inputs = Input(shape=(config['time_steps'], 1))
    lstm1 = LSTM(100, return_sequences=True)(inputs)
    lstm2 = LSTM(100)(lstm1)
    dense1 = Dense(20, activation='relu')(lstm2)
    dense2 = Dense(n_qubits, activation='relu')(dense1)

    # VQC layer, reshape the output to remove the complex part
    quantum_layer = qml.qnn.KerasLayer(quantum_circuit, weight_shapes, output_dim=1)(dense2)
    quantum_layer = tf.reshape(quantum_layer, (-1, 1))

    dense3 = Dense(n_qubits, activation='relu')(quantum_layer)
    output = Dense(config['future_steps'], activation='linear')(dense3)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=config['learning_rate']), loss=config['loss_function'])
    return model


def create_ph_quantum_circuit(n_qubits, n_layers, config):
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev, interface='tf')
    def quantum_circuit(inputs, weights):
        qml.AngleEmbedding(inputs, wires=range(n_qubits))
        qml.StronglyEntanglingLayers(weights, wires=range(n_qubits))
        return qml.expval(qml.PauliZ(range(n_qubits)))

    weight_shapes = {"weights": (n_layers, n_qubits, 3)}
    return quantum_circuit, weight_shapes
