from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, LSTM, Dense, Dropout,
    Embedding, Flatten, Concatenate
)

def create_emotion_conditioned_model(note_input_shape, emotion_vocab_size, note_vocab_size):
    """
    note_input_shape  : (SEQUENCE_LENGTH, 1)
    emotion_vocab_size: number of emotions (5)
    note_vocab_size   : total unique notes
    """

    # =========================
    # NOTE INPUT (MUSIC)
    # =========================
    notes_input = Input(shape=note_input_shape, name="notes_input")
    x = LSTM(128, return_sequences=True)(notes_input)
    x = Dropout(0.3)(x)
    x = LSTM(128)(x)

    # =========================
    # EMOTION INPUT
    # =========================
    emotion_input = Input(shape=(1,), name="emotion_input")
    e = Embedding(emotion_vocab_size, 8)(emotion_input)
    e = Flatten()(e)
    e = Dense(128, activation="relu")(e)

    # =========================
    # MERGE MUSIC + EMOTION
    # =========================
    combined = Concatenate()([x, e])
    combined = Dense(256, activation="relu")(combined)

    output = Dense(note_vocab_size, activation="softmax")(combined)

    model = Model(
        inputs=[notes_input, emotion_input],
        outputs=output
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer="adam"
    )

    return model
