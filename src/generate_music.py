import numpy as np
import random
from tensorflow.keras.models import load_model
from music21 import note, chord, stream

from data_preprocessing import load_midi_files, prepare_sequences


def generate_music():
    # 1. Load notes and sequences
    notes = load_midi_files("data/midi", max_notes=15000)
    X, _, pitchnames = prepare_sequences(notes)

    n_vocab = len(pitchnames)

    # 2. Load trained model
    model = load_model("best_model.h5")

    # 3. Pick a random seed sequence
    start_index = random.randint(0, len(X) - 1)
    pattern = X[start_index]

    int_to_note = dict((i, note) for i, note in enumerate(pitchnames))

    prediction_output = []

    # 4. Generate notes
    for _ in range(200):  # number of notes to generate
        prediction_input = np.reshape(pattern, (1, pattern.shape[0], pattern.shape[1]))
        prediction = model.predict(prediction_input, verbose=0)

        index = np.argmax(prediction)
        result = int_to_note[index]
        prediction_output.append(result)

        # shift pattern
        pattern = np.append(pattern[1:], [[index]], axis=0)

    # 5. Convert to MIDI
    offset = 0
    output_notes = []

    for pattern in prediction_output:
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes_list = []

            for current_note in notes_in_chord:
                new_note = note.Note(int(current_note))
                new_note.storedInstrument = None
                notes_list.append(new_note)

            new_chord = chord.Chord(notes_list)
            new_chord.offset = offset
            output_notes.append(new_chord)

        else:
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = None
            output_notes.append(new_note)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)
    midi_stream.write("midi", fp="output/generated_music.mid")


if __name__ == "__main__":
    generate_music()
