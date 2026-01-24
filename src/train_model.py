import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint

from data_preprocessing import load_midi_files, prepare_sequences
from model import create_model


def train():
    # 1. Load and preprocess data
    notes = load_midi_files("data/midi", max_notes=15000)
    X, y, pitchnames = prepare_sequences(notes)

    n_vocab = len(pitchnames)

    # 2. Convert output labels to categorical
    y = to_categorical(y, num_classes=n_vocab)

    # 3. Create model
    model = create_model(
        input_shape=(X.shape[1], X.shape[2]),
        n_vocab=n_vocab
    )

    # 4. Save best model during training
    checkpoint = ModelCheckpoint(
        "best_model.h5",
        monitor="loss",
        save_best_only=True,
        mode="min"
    )

    # 5. Train the model
    model.fit(
        X,
        y,
        epochs=30,
        batch_size=64,
        callbacks=[checkpoint]
    )


if __name__ == "__main__":
    train()
