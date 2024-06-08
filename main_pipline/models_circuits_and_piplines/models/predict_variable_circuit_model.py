from keras.models import Model
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.optimizers import Adam
from main_pipline.input.div.logger import logger
import pennylane as qml
import tensorflow as tf
from tqdm import tqdm

# Perhaps TODO adapt Model to work with circuits if needed
class PVCModel:
    def __init__(self, circuit, config):
        self.config = config
        self.model = create_pvc_model(circuit, config)

    def train(self, x_train, y_train):
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
            logger.info(f"Epoch {epoch+1}/{epochs} loss: {epoch_loss / steps_per_epoch}")
            tqdm.write(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / steps_per_epoch}")
        return history

    def evaluate(self, x_test, y_test):
        loss = self.model.evaluate(x_test, y_test)
        return loss

    def predict(self, x):
        return self.model.predict(x)


def create_pvc_model(quantum_circuit, config):
    inputs = Input(shape=(config['time_steps'], config['input_dim']))
    dense1 = Dense(config['n_qubits'], activation='linear')(inputs)
    quantum_layer = qml.qnn.KerasLayer(quantum_circuit, output_dim=1)(dense1)
    quantum_layer = tf.reshape(quantum_layer, (-1, 1))
    output = Dense(config['future_steps'], activation='linear')(quantum_layer)

    model = Model(inputs=inputs, outputs=output)
    model.compile(optimizer=Adam(learning_rate=config['learning_rate']), loss=config['loss_function'])
    return model
