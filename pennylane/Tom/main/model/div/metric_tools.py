import matplotlib.pyplot as plt
import tensorflow as tf

def plot_results(x_test, y_test, x_train, y_train, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(x_test.numpy(), y_test, label='True Function', color='blue')
    plt.plot(x_test.numpy(), y_pred, label='VQC Approximation', color='red')
    plt.scatter(x_train.numpy(), y_train.numpy(), color='green', label='Training Points')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Quantum Circuit Learning Complex Function using PennyLane and TensorFlow')
    plt.legend()
    plt.grid(True)
    plt.show()


class TrainingPlot(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_epoch_end(self, epoch, logs={}):
        self.losses.append(logs.get('loss'))
        if epoch % 10 == 0:  # Plot every 10 epochs
            plt.figure()
            plt.plot(range(len(self.losses)), self.losses, label='Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss')
            plt.title('Training Loss')
            plt.legend()
            plt.show()

