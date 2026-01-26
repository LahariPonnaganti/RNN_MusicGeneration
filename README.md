# RNN Music Generation

## Overview

This project implements an **Automatic Music Generation system** using **Recurrent Neural Networks (RNNs)** with **Long Short-Term Memory (LSTM)**.

The model learns sequential musical patterns from piano MIDI files and generates new music automatically.  
Since music is inherently time-dependent, RNN–LSTM architectures are suitable for capturing pitch progression, rhythm, tempo, and note dependencies.
To improve musical diversity and reduce repetition, the system uses temperature-based probabilistic sampling during music generation.

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
│   └── generate_music.py      # Music generation using temperature-based sampling
├── output/
│   └── generated_music.mid    # Generated music output
├── best_model.h5              # Trained model weights
├── requirements.txt           # Required Python libraries
└── README.md
```

---

## How to Use / Run the Project

Follow these steps to run the project on your local machine.  
During generation, the next musical note is selected using temperature-based probability sampling instead of greedy prediction.

### Step 1: Clone the Repository

```bash
git clone https://github.com/LahariPonnaganti/RNN_MusicGeneration.git
cd RNN_MusicGeneration
```

### Step 2: Create and Activate Virtual Environment (Windows)

```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Required Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Generate Music

```bash
python src/generate_music.py
```

### Step 5: Output

The generated music file will be saved at:

```text
output/generated_music.mid

```
