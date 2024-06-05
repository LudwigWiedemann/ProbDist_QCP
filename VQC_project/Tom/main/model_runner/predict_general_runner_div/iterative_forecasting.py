def iterative_forecast(model, dataset, config):
    steps = config['steps_to_predict']
    time_steps = config['time_steps']
    future_steps = config['future_steps']

    initial_input = dataset['Future_Input']
    # Start of prediction
    current_input = initial_input
    all_predictions = []

    for _ in range(steps // future_steps):
        # Ensure the current input has the correct shape
        if current_input.shape[1] < time_steps:
            padding = np.zeros((current_input.shape[0], time_steps - current_input.shape[1], current_input.shape[2]))
            current_input = np.concatenate((padding, current_input), axis=1)

        pred = model.predict(current_input)
        all_predictions.append(pred.flatten())
        # Use the last `time_steps` of the combined current_input + prediction as the next input
        current_input = np.concatenate((current_input.flatten(), pred.flatten()))[-time_steps:].reshape(1, -1, 1)
    print_iterative_forcast_plot(function, initial_input, np.concatenate(all_predictions), config)

