import matplotlib.pyplot as plt


def plot_metrics(history):
    # Plot the training loss and validation loss if available
    plt.figure()
    plt.plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.show()


def plot_predictions(x_data, y_real, y_pred, title='Real vs Predicted', confidence_interval=None):
    plt.figure()
    plt.plot(x_data, y_real, label='Real', color='blue')
    plt.plot(x_data, y_pred, label='Predicted', color='red')

    if confidence_interval is not None:
        lower_bound, upper_bound = confidence_interval
        plt.fill_between(x_data, lower_bound, upper_bound, color='red', alpha=0.2)

    plt.xlabel('Time Steps')
    plt.ylabel('Values')
    plt.title(title)
    plt.legend()
    plt.show()


def plot_residuals(y_real, y_pred, title='Residuals'):
    residuals = y_real - y_pred
    plt.figure()
    plt.plot(residuals, label='Residuals', color='purple')
    plt.axhline(0, color='black', linestyle='--')
    plt.xlabel('Samples')
    plt.ylabel('Residuals')
    plt.title(title)
    plt.legend()
    plt.show()