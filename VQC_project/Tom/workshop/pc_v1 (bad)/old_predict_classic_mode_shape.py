from keras import Model
from tensorflow.keras.layers import Dense, Input, LSTM
from tensorflow.keras.optimizers import Adam


def create_pc_model(config):
    inputs = Input(shape=(config['time_steps'], config['input_dim']))
    lstm1 = LSTM(50, return_sequences=True)(inputs)
    lstm2 = LSTM(50)(lstm1)
    dense1 = Dense(10, activation='relu')(lstm2)
    dense2 = Dense(config['future_steps'], activation='linear')(dense1)
    model = Model(inputs=inputs, outputs=dense2)
    model.compile(optimizer=Adam(learning_rate=config['learning_rate']), loss=config['loss_function'])
    return model