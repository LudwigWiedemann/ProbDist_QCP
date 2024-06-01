import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from tensorflow.keras.callbacks import Callback
from tqdm import tqdm
import numpy as np

def plot_pc_results(x_test, y_test, x_train, y_train, y_pred):
    plt.figure(figsize=(10, 6))
    plt.plot(x_test.numpy(), y_test, label='True Function', color='blue')
    plt.plot(x_test.numpy(), y_pred, label='Model Prediction', color='red')

    x_train_flat = x_train.numpy().reshape(-1)
    y_train_flat = np.repeat(y_train.numpy(), x_train.shape[1])

    plt.scatter(x_train_flat, y_train_flat, color='green', label='Training Points')

    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Model Prediction vs True Function')
    plt.legend()
    plt.grid(True)
    plt.show()

class Pc_TrainingPlot(Callback):
    def __init__(self, x_train, y_train, x_test, y_test, interval=1):
        super(Pc_TrainingPlot, self).__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.interval = interval  # Interval (epochs) to update the plot
        self.epoch_count = 0
        self.fig, self.ax = plt.subplots(figsize=(10, 6))
        self.train_scatter, = self.ax.plot([], [], 'go', label='Training Points')
        self.test_line, = self.ax.plot([], [], 'b-', label='True Function')
        self.pred_line, = self.ax.plot([], [], 'r-', label='Model Prediction')
        self.ax.set_xlabel('x')
        self.ax.set_ylabel('f(x)')
        self.ax.set_title('Training Progress')
        self.ax.legend()
        self.ax.grid(True)

        self.anim = FuncAnimation(self.fig, self.update_plot, init_func=self.init_plot, interval=200, blit=True)
        plt.ion()
        plt.show()

    def init_plot(self):
        self.train_scatter.set_data([], [])
        self.test_line.set_data([], [])
        self.pred_line.set_data([], [])
        return self.train_scatter, self.test_line, self.pred_line

    def update_plot(self, frame):
        y_pred = self.model.predict(self.x_test)

        self.train_scatter.set_data(self.x_train.numpy().flatten(), np.repeat(self.y_train.numpy(), self.x_train.shape[1]))
        self.test_line.set_data(self.x_test.numpy().flatten(), self.y_test)
        self.pred_line.set_data(self.x_test.numpy().flatten(), y_pred.flatten())

        self.ax.relim()
        self.ax.autoscale_view()

        return self.train_scatter, self.test_line, self.pred_line

    def on_epoch_end(self, epoch, logs=None):
        self.epoch_count += 1
        if self.epoch_count % self.interval == 0:
            print(f"Updated plot at epoch {epoch}")
        if self.tqdm_bar:
            self.tqdm_bar.set_postfix(logs)
            self.tqdm_bar.update(1)

    def on_train_begin(self, logs=None):
        self.tqdm_bar = tqdm(total=self.params['epochs'], desc='Training Progress', unit='epoch')

    def on_train_end(self, logs=None):
        if self.tqdm_bar:
            self.tqdm_bar.close()
        plt.ioff()
        plt.show()
