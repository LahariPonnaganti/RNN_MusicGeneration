import os
import pickle
import numpy as np
from music21 import converter, instrument, note, chord

DATA_DIR = "data/midi"
SEQUENCE_LENGTH = 50
OUTPUT_DIR = "data"

EMOTION_MAP = {
    "happy": 0,
    "sad": 1,
    "calm": 2,
    "energetic": 3,
    "angry": 4
}

all_notes = []
all_emotions = []

print("🔄 Starting preprocessing...")

# =========================
# READ MIDI FILES
# =========================
for emotion in os.listdir(DATA_DIR):
    emotion_path = os.path.join(DATA_DIR, emotion)

    if emotion not in EMOTION_MAP:
        continue

    for file in os.listdir(emotion_path):
        if not file.endswith(".mid"):
            continue

        path = os.path.join(emotion_path, file)
        print(f"Processing: {path}")

        midi = converter.parse(path)
        parts = instrument.partitionByInstrument(midi)
        elements = parts.parts[0].recurse() if parts else midi.flat.notes

        for el in elements:
            if isinstance(el, note.Note):
                all_notes.append(str(el.pitch.midi))
                all_emotions.append(EMOTION_MAP[emotion])

            elif isinstance(el, chord.Chord):
                all_notes.append(".".join(str(n.pitch.midi) for n in el.notes))
                all_emotions.append(EMOTION_MAP[emotion])

print(f"Total notes collected: {len(all_notes)}")

# =========================
# CREATE MAPPINGS
# =========================
unique_notes = sorted(set(all_notes))
note_to_int = {note: i for i, note in enumerate(unique_notes)}

os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(os.path.join(OUTPUT_DIR, "note_to_int.pkl"), "wb") as f:
    pickle.dump(note_to_int, f)

print("✅ note_to_int.pkl saved successfully")

# =========================
# PREPARE SEQUENCES
# =========================
network_input = []
network_output = []
emotion_labels = []

for i in range(len(all_notes) - SEQUENCE_LENGTH):
    seq_in = all_notes[i:i + SEQUENCE_LENGTH]
    seq_out = all_notes[i + SEQUENCE_LENGTH]

    network_input.append([note_to_int[n] for n in seq_in])
    network_output.append(note_to_int[seq_out])
    emotion_labels.append(all_emotions[i])

network_input = np.array(network_input)
network_output = np.array(network_output)
emotion_labels = np.array(emotion_labels)

np.save(os.path.join(OUTPUT_DIR, "network_input.npy"), network_input)
np.save(os.path.join(OUTPUT_DIR, "network_output.npy"), network_output)
np.save(os.path.join(OUTPUT_DIR, "emotion_labels.npy"), emotion_labels)

print("✅ Preprocessing complete")
