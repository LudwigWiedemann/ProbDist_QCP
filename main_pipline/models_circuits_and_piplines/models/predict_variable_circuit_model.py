from keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.optimizers import Adam
from main_pipline.input.div.logger import logger
import pennylane as qml
import tensorflow as tf
from tqdm import tqdm


class PVCModel:
    def __init__(self, variable_circuit, config):
        self.config = config
        self.circuit = variable_circuit()
        self.model = self.create_pvc_model(self.circuit, config)

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
            logger.info(f"Epoch {epoch + 1}/{epochs} loss: {epoch_loss / steps_per_epoch}")
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

    def create_pvc_model(self, circuit, config):
        inputs = Input(shape=(config['time_steps'], 1))
        reshaped_inputs = tf.keras.layers.Reshape((config['time_steps'],))(inputs)
        quantum_layer = qml.qnn.KerasLayer(circuit.run(), circuit.get_weights(), output_dim=1)(reshaped_inputs)
        model = Model(inputs=inputs, outputs=quantum_layer)
        model.compile(optimizer=Adam(learning_rate=config['learning_rate']), loss=config['loss_function'])
        return model