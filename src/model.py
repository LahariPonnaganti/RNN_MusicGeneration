from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
def create_model(input_shape, n_vocab):
    model = Sequential()

    model.add(
        LSTM(
            256,
            input_shape=input_shape,
            return_sequences=True
        )
    )
    model.add(Dropout(0.3))

    model.add(LSTM(256))
    model.add(Dropout(0.3))

    model.add(Dense(n_vocab, activation='softmax'))

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam'
    )

    return model
