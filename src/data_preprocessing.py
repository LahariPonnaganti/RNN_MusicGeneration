import os
import numpy as np
from music21 import converter, note, chord


def load_midi_files(midi_folder, max_notes=15000):
    notes = []

    for file in os.listdir(midi_folder):
        if file.endswith(".mid"):
            file_path = os.path.join(midi_folder, file)
            midi = converter.parse(file_path)

            for element in midi.flat.notes:
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))

                if len(notes) >= max_notes:
                    return notes

    return notes


def prepare_sequences(notes, sequence_length=50):
    pitchnames = sorted(set(notes))
    note_to_int = {note: number for number, note in enumerate(pitchnames)}

    network_input = []
    network_output = []

    for i in range(len(notes) - sequence_length):
        seq_in = notes[i:i + sequence_length]
        seq_out = notes[i + sequence_length]

        network_input.append([note_to_int[n] for n in seq_in])
        network_output.append(note_to_int[seq_out])

    network_input = np.reshape(
        network_input,
        (len(network_input), sequence_length, 1)
    )

    network_input = network_input / float(len(pitchnames))

    return network_input, np.array(network_output), pitchnames


if __name__ == "__main__":
    notes = load_midi_files("data/midi", max_notes=15000)
    X, y, pitchnames = prepare_sequences(notes)

    print("Total notes used:", len(notes))
    print("Unique notes:", len(pitchnames))
    print("Input shape:", X.shape)
    print("Output shape:", y.shape)
