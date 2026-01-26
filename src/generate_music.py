import os
import numpy as np
import random
from tensorflow.keras.models import load_model
from music21 import note, chord, stream

from data_preprocessing import load_midi_files, prepare_sequences


def sample_with_temperature(preds, temperature=1.0):
    preds = np.asarray(preds).astype("float64")
    preds = np.log(preds + 1e-8) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    return np.random.choice(len(preds), p=preds)


def generate_music(
    midi_dir="data/midi",
    model_path="best_model.h5",
    output_path="output/generated_music.mid",
    n_generate=200,
    temperature=0.8,
):
    # -----------------------------
    # 0. Ensure output directory
    # -----------------------------
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # -----------------------------
    # 1. Load data
    # -----------------------------
    notes = load_midi_files(midi_dir, max_notes=15000)
    X, _, pitchnames = prepare_sequences(notes)

    vocab_size = len(pitchnames)

    # -----------------------------
    # 2. Load trained model
    # -----------------------------
    model = load_model(model_path)

    # -----------------------------
    # 3. Pick random seed
    # -----------------------------
    start_index = random.randint(0, len(X) - 1)
    pattern = X[start_index].copy()

    print("Seed pattern shape:", pattern.shape)  # should be (seq_len, 1)

    int_to_note = {i: str(n) for i, n in enumerate(pitchnames)}
    prediction_output = []

    # -----------------------------
    # 4. Generate notes
    # -----------------------------
    for _ in range(n_generate):
        prediction_input = pattern.reshape(
            1, pattern.shape[0], pattern.shape[1]
        )

        prediction = model.predict(prediction_input, verbose=0)[0]

        index = sample_with_temperature(prediction, temperature)
        index = index % vocab_size

        prediction_output.append(int_to_note[index])

        # integer sliding window (IMPORTANT)
        pattern = np.append(pattern[1:], [[index]], axis=0)

    # -----------------------------
    # 5. Convert to MIDI
    # -----------------------------
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        try:
            if "." in pattern:
                notes_in_chord = pattern.split(".")
                notes_list = [note.Note(int(n)) for n in notes_in_chord]
                new_chord = chord.Chord(notes_list)
                new_chord.offset = offset
                output_notes.append(new_chord)
            else:
                new_note = note.Note(pattern)
                new_note.offset = offset
                output_notes.append(new_note)

            offset += 0.5

        except Exception:
            # skip invalid notes safely
            continue

    midi_stream = stream.Stream(output_notes)
    midi_stream.write("midi", fp=output_path)

    print("🎵 Music generated successfully:", output_path)


if __name__ == "__main__":
    generate_music()
