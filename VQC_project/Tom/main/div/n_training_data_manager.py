import numpy as np


def generate_time_series_data(config, func):
    """
    Generate a more complex sample set based on a given mathematical function.
    :param config: Configuration dictionary
    :param func: A mathematical function to generate the data (e.g., np.sin)
    :return: Tuple of training, testing, and future datasets (x_train, y_train, x_test, y_test, x_future)
    """
    time_steps = config['time_steps']
    num_samples = config['num_samples']
    future_steps = config['future_steps']
    noise_level = config.get('noise_level', 0.0)
    data_length = config['data_length']

    # Ensure the length of x_data is greater than time_steps
    data_length = max(time_steps + future_steps, data_length)  # Ensure a minimum length
    x_data = np.linspace(0, 4 * np.pi, data_length)  # Extended range for more future data
    y_data = func(x_data)

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    if time_steps + future_steps >= len(x_data):
        raise ValueError("time_steps + future_steps must be less than the length of x_data")

    for _ in range(num_samples):
        start_idx = np.random.randint(0, len(x_data) - time_steps - future_steps)
        x_sample = y_data[start_idx:start_idx + time_steps]
        y_sample = y_data[start_idx + time_steps:start_idx + time_steps + future_steps]

        # Add noise to the samples
        x_sample += np.random.normal(0, noise_level, x_sample.shape)

        if np.random.rand() < 0.8:
            x_train.append(x_sample)
            y_train.append(y_sample)
        else:
            x_test.append(x_sample)
            y_test.append(y_sample)

    # Prepare future data for prediction
    future_start_idx = len(x_data) - time_steps
    x_future = y_data[future_start_idx:future_start_idx + time_steps]

    x_train = np.array(x_train).reshape(-1, time_steps, 1)
    y_train = np.array(y_train).reshape(-1, future_steps)
    x_test = np.array(x_test).reshape(-1, time_steps, 1)
    y_test = np.array(y_test).reshape(-1, future_steps)
    x_future = np.array(x_future).reshape(1, time_steps, 1)  # Single future sequence

    return x_train, y_train, x_test, y_test, x_future


# Example of usage
if __name__ == "__main__":
    config = {
        "time_steps": 50,
        "input_dim": 1,
        "num_samples": 1000,
        "noise_level": 0.1,  # Adjust noise level as needed
        "future_steps": 50  # Number of future steps for prediction
    }

    x_train, y_train, x_test, y_test, x_future = generate_time_series_data(config, np.sin)
    print("Generated time series data:")
    print(f"x_train shape: {x_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"x_test shape: {x_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    print(f"x_future shape: {x_future.shape}")
