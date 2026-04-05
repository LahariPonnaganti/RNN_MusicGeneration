import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from model import create_emotion_conditioned_model

SEQUENCE_LENGTH = 50

# =========================
# LOAD DATA
# =========================
X_notes = np.load("data/network_input.npy")
y_notes = np.load("data/network_output.npy")
emotion_labels = np.load("data/emotion_labels.npy")

n_vocab = int(np.max(X_notes)) + 1

X_notes = np.reshape(X_notes, (X_notes.shape[0], SEQUENCE_LENGTH, 1))
X_notes = X_notes / float(n_vocab)

y_notes = to_categorical(y_notes, num_classes=n_vocab)

# =========================
# BUILD MODEL
# =========================
model = create_emotion_conditioned_model(
    note_input_shape=(SEQUENCE_LENGTH, 1),
    emotion_vocab_size=5,
    note_vocab_size=n_vocab
)

# =========================
# TRAIN
# =========================
checkpoint = ModelCheckpoint(
    "model/emotion_music_model.h5",
    monitor="loss",
    save_best_only=True
)

model.fit(
    [X_notes, emotion_labels],
    y_notes,
    epochs=50,
    batch_size=64,
    callbacks=[checkpoint]
)

print("✅ Model training complete")
