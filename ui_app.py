import streamlit as st
import subprocess
import os
import sys

# make src accessible
sys.path.append("src")

from generate_music import generate_from_user_midi

# -------------------------
# CONFIG
# -------------------------
SOUNDFONT_PATH = "FluidR3_GM.sf2"
OUTPUT_DIR = "output"
WAV_PATH = os.path.join(OUTPUT_DIR, "generated.wav")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# UI
# -------------------------
st.title("🎵 Emotion-Conditioned Music Generator")

uploaded_file = st.file_uploader("Upload MIDI file (seed)", type=["mid", "midi"])

emotion = st.selectbox(
    "Select Emotion",
    ["happy", "sad", "calm", "energetic", "angry"]
)

length = st.slider("Generation Length (notes)", 50, 500, 200)

# -------------------------
# GENERATE
# -------------------------
if st.button("Generate Music"):

    if uploaded_file is None:
        st.error("Please upload a MIDI file.")
        st.stop()

    # Save uploaded MIDI
    seed_path = os.path.join("temp_seed.mid")
    with open(seed_path, "wb") as f:
        f.write(uploaded_file.read())

    with st.spinner("Generating music..."):
        # 1️⃣ Generate MIDI
        midi_path = generate_from_user_midi(
            input_midi_path=seed_path,
            emotion=emotion,
            generate_length=length
        )

        # 2️⃣ Convert MIDI → WAV using FluidSynth
        subprocess.run(
            [
                "fluidsynth",
                "-ni",
                "-r", "44100",
                "-F", WAV_PATH,
                SOUNDFONT_PATH,
                midi_path
            ],
            check=True
        )

    st.success("Music generated!")

    # 3️⃣ Auto-play audio
    with open(WAV_PATH, "rb") as audio_file:
        st.audio(audio_file.read(), format="audio/wav")
