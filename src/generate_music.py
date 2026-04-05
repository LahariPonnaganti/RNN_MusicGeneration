import os
import numpy as np
import pickle

from music21 import converter, instrument, note, chord, stream
from tensorflow.keras.models import load_model

# =========================
# PATHS
# =========================
MODEL_PATH = "model/emotion_music_model.h5"
MAPPING_PATH = "data/note_to_int.pkl"
OUTPUT_DIR = "output"

SEQUENCE_LENGTH = 50

EMOTION_MAP = {
    "happy": 0,
    "sad": 1,
    "calm": 2,
    "energetic": 3,
    "angry": 4
}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# LOAD MODEL + MAPPINGS
# =========================
model = load_model(MODEL_PATH)

with open(MAPPING_PATH, "rb") as f:
    note_to_int = pickle.load(f)

int_to_note = {v: k for k, v in note_to_int.items()}
VOCAB_SIZE = len(note_to_int)

# =========================
# EXTRACT NOTES FROM MIDI
# =========================
def extract_notes_from_midi(midi_path):
    midi = converter.parse(midi_path)
    notes = []

    parts = instrument.partitionByInstrument(midi)
    elements = parts.parts[0].recurse() if parts else midi.flat.notes

    for el in elements:
        if isinstance(el, note.Note):
            notes.append(str(el.pitch.midi))
        elif isinstance(el, chord.Chord):
            notes.append(".".join(str(n.pitch.midi) for n in el.notes))

    return notes

# =========================
# PREPARE SEED
# =========================
def prepare_seed(notes):
    seed = [note_to_int[n] for n in notes if n in note_to_int]

    if len(seed) < SEQUENCE_LENGTH:
        raise ValueError("Uploaded MIDI too short or incompatible.")

    return seed[:SEQUENCE_LENGTH]

# =========================
# GENERATE MUSIC
# =========================
def generate_from_user_midi(input_midi_path, emotion, generate_length=200):
    emotion_id = EMOTION_MAP[emotion]

    # ---------- Seed ----------
    if input_midi_path:
        raw_notes = extract_notes_from_midi(input_midi_path)
        pattern = prepare_seed(raw_notes)
    else:
        # fallback random seed
        pattern = np.random.randint(0, VOCAB_SIZE, SEQUENCE_LENGTH).tolist()

    generated = []

    # ---------- Generation ----------
    for _ in range(generate_length):
        x = np.reshape(pattern, (1, SEQUENCE_LENGTH, 1)) / float(VOCAB_SIZE)
        e = np.array([[emotion_id]])

        prediction = model.predict([x, e], verbose=0)
        index = np.argmax(prediction)

        generated.append(index)
        pattern.append(index)
        pattern = pattern[1:]

    # ---------- Convert to MIDI ----------
    output_notes = []
    offset = 0

    for idx in generated:
        token = int_to_note[idx]

        if "." in token:
            chord_notes = []
            for n in token.split("."):
                pitch = int(n)
                if 21 <= pitch <= 108:
                    chord_notes.append(note.Note(pitch))
            if chord_notes:
                c = chord.Chord(chord_notes)
                c.offset = offset
                output_notes.append(c)
        else:
            pitch = int(token)
            if 21 <= pitch <= 108:
                n = note.Note(pitch)
                n.offset = offset
                output_notes.append(n)

        offset += 0.5

    midi_stream = stream.Stream(output_notes)

    output_path = os.path.join(
        OUTPUT_DIR, f"generated_{emotion}.mid"
    )
    midi_stream.write("midi", output_path)

    return output_path
