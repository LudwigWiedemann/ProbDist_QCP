from keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.optimizers import Adam
import pennylane as qml
import tensorflow as tf
from tqdm import tqdm


class PACModel:
    def __init__(self, variable_circuit, config):
        self.config = config
        self.circuit = variable_circuit(config)
        self.model = self.create_pac_model(self.circuit, config)
        self.optimizer = Adam(learning_rate=config['learning_rate'])
        self.loss_fn = tf.keras.losses.get(config['loss_function'])
        self.normalization_factor = None

    def train(self, dataset, logger):
        x_train = dataset['input_train'] / self.config['compress_factor']
        y_train = dataset['output_train'] / self.config['compress_factor']

        epochs = self.config['epochs']
        batch_size = self.config['batch_size']
        steps_per_epoch = len(x_train) // batch_size

        history = {'loss': []}

        # Early stopping parameters
        patience = self.config['patience']
        min_delta = self.config['min_delta']
        best_loss = float('inf')
        wait = 0

        for epoch in tqdm(range(epochs), desc="Training Progress"):
            epoch_loss = 0
            for step in range(steps_per_epoch):
                batch_start = step * batch_size
                batch_end = (step + 1) * batch_size
                x_batch = x_train[batch_start:batch_end]
                y_batch = y_train[batch_start:batch_end]

                batch_loss = self.model.train_on_batch(x_batch, y_batch)
                epoch_loss += batch_loss

            epoch_loss /= steps_per_epoch
            if self.normalization_factor is None:
                self.normalization_factor = epoch_loss  # Initialize normalization factor with the first epoch loss
            normalized_loss = epoch_loss / self.normalization_factor  # Normalize the loss
            history['loss'].append(normalized_loss)

            log_message = f"Epoch {epoch + 1}/{epochs} loss: {normalized_loss}"
            logger.info(log_message)  # Log using the logger
            tqdm.write(f"Epoch {epoch + 1}/{epochs}, Loss: {normalized_loss}")  # Print using tqdm

            # Early stopping check
            if normalized_loss < best_loss - min_delta:
                best_loss = normalized_loss
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    logger.info(f"Early stopping at epoch {epoch + 1}")
                    break

        return history

    def evaluate(self, dataset):
        x_test = dataset['input_test'] / self.config['compress_factor']
        y_test = dataset['output_test'] / self.config['compress_factor']

        evaluation_results = self.model.evaluate(x_test, y_test, return_dict=True)
        predictions = self.model.predict(x_test) * self.config['compress_factor']

        normalized_loss = evaluation_results['loss'] / self.normalization_factor  # Normalize the loss
        return predictions, normalized_loss

    def predict(self, x_test):
        return self.model.predict((x_test / self.config['compress_factor'])) * self.config['compress_factor']

    def create_pac_model(self, circuit, config):
        inputs = Input(shape=(config['time_steps'], 1))
        reshaped_inputs = tf.keras.layers.Reshape((config['time_steps'],))(inputs)
        quantum_layer = qml.qnn.KerasLayer(circuit.run(), circuit.get_weights(), output_dim=config['time_steps'])(reshaped_inputs)
        model = Model(inputs=inputs, outputs=quantum_layer)
        model.compile(optimizer=Adam(learning_rate=config['learning_rate']), loss=config['loss_function'])
        return model

    def save_model(self, path, logger):
        self.model.save_weights(path, overwrite=True)
        logger.info(f"Model saved to {path}")

    def load_model(self, path, logger):
        self.model = tf.keras.models.load_model(path, custom_objects={'KerasLayer': qml.qnn.KerasLayer})
        logger.info(f"Model loaded from {path}")
