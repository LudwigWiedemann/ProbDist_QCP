from keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
from main_pipline.input.div.logger import logger
import pennylane as qml
import tensorflow as tf
from tqdm import tqdm


class PACModel:
    def __init__(self, variable_circuit, config):
        self.config = config
        self.circuit = variable_circuit(config)
        self.model, self.scaling_factor = self.create_pac_model(self.circuit, config)

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
            scaling_value = self.scaling_factor.numpy()
            logger.info(
                f"Epoch {epoch + 1}/{epochs} loss: {epoch_loss / steps_per_epoch}, scaling factor: {scaling_value}")
            tqdm.write(
                f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss / steps_per_epoch}, Scaling Factor: {scaling_value}")
        return history

    def evaluate(self, dataset):
        x_test = dataset['input_test']
        y_test = dataset['output_test']

        pred_y_test_data = self.model.predict(x_test)
        loss = self.model.evaluate(x_test, y_test)
        return pred_y_test_data, loss

    def predict(self, x_test):
        return self.model.predict(x_test)

    def create_pac_model(self, circuit, config):
        inputs = Input(shape=(config['time_steps'], 1))
        reshaped_inputs = tf.keras.layers.Reshape((config['time_steps'],))(inputs)
        quantum_layer = qml.qnn.KerasLayer(circuit.run(), circuit.get_weights(), output_dim=config['time_steps'])(
            reshaped_inputs)

        scaling_factor = tf.Variable(initial_value=2.0, trainable=True, dtype=tf.float64, name="scaling_factor")
        outputs = tf.cast(quantum_layer, tf.float64) ** scaling_factor

        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=Adam(learning_rate=config['learning_rate']), loss=config['loss_function'])
        return model, scaling_factor

    def save_model(self, path):
        self.model.save(path)
        logger.info(f"Model saved to {path}")

    def load_model(self, path):
        self.model = tf.keras.models.load_model(path, custom_objects={'KerasLayer': qml.qnn.KerasLayer})
        logger.info(f"Model loaded from {path}")
