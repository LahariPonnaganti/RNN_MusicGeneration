# RNN Music Generation

## Overview

This project implements an **Automatic Music Generation system** using **Recurrent Neural Networks (RNNs)** with **Long Short-Term Memory (LSTM)**.

The model learns sequential musical patterns from piano MIDI files and generates new music automatically.  
Since music is inherently time-dependent, RNN–LSTM architectures are suitable for capturing pitch progression, rhythm, tempo, and note dependencies.

---

## Objectives

- To generate music automatically using deep learning techniques
- To learn sequential musical patterns such as pitch and rhythm
- To demonstrate the application of RNN–LSTM models in music generation

---

## Technologies Used

- Python 3.10
- TensorFlow / Keras
- NumPy
- music21
- MIDI (Musical Instrument Digital Interface)

---

## Project Structure

```text
RNN_MusicGeneration/
├── data/
│   └── midi/                  # Input piano MIDI dataset
├── src/
│   ├── data_preprocessing.py  # Converts MIDI files to note sequences
│   ├── model.py               # LSTM model architecture
│   ├── train_model.py         # Model training script
│   └── generate_music.py      # Music generation script
├── output/
│   └── generated_music.mid    # Generated music output
├── best_model.h5              # Trained model weights
├── requirements.txt           # Required Python libraries
└── README.md
```
