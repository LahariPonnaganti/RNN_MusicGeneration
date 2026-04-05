# 🎵 RNN Music Generation using LSTM

## 📌 Overview

This project implements an **Automatic Music Generation system** using **Recurrent Neural Networks (RNNs)** with **Long Short-Term Memory (LSTM)**.

The model learns sequential musical patterns from MIDI files and generates new music automatically.
Since music is inherently time-dependent, RNN–LSTM architectures are well-suited for capturing pitch progression, rhythm, tempo, and note dependencies.

To improve musical diversity and reduce repetition, the system uses **temperature-based probabilistic sampling** during music generation.

---

## 🎯 Objectives

* Generate music automatically using deep learning techniques
* Learn sequential musical patterns such as pitch and rhythm
* Demonstrate the application of RNN–LSTM models in music generation

---

## 🛠 Technologies Used

* Python 3.10
* TensorFlow / Keras
* NumPy
* music21
* MIDI (Musical Instrument Digital Interface)

---

## 📁 Project Structure

```text
RNN_MusicGeneration/
├── src/
│   ├── data_preprocessing.py   # Converts MIDI files to note sequences
│   ├── model.py                # LSTM model architecture
│   ├── train_model.py          # Model training script
│   └── generate_music.py       # Music generation using temperature sampling
├── ui_app.py                   # Optional UI for music generation
├── requirements.txt            # Required Python libraries
└── README.md
```

---

## 📂 Dataset

The dataset is **not included** in this repository due to size limitations.

You need to provide your own MIDI dataset organized as:

```text
data/midi/<emotion>/*.mid
```

Example:

```text
data/midi/happy/
data/midi/sad/
data/midi/calm/
data/midi/energetic/
data/midi/angry/
```

---

## 🤖 Model

The trained model file (`.h5`) is **not included** due to size constraints.

You can:

* Train the model from scratch
* Or use your own pretrained model

---

## ▶️ How to Run the Project

### Step 1: Clone the Repository

```bash
git clone https://github.com/LahariPonnaganti/RNN_MusicGeneration.git
cd RNN_MusicGeneration
```

---

### Step 2: Create and Activate Virtual Environment (Windows)

```bash
python -m venv venv
venv\Scripts\activate
```

---

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

---

### Step 4: Train the Model

```bash
python src/train_model.py
```

---

### Step 5: Generate Music

```bash
python src/generate_music.py
```

---

## 🎧 Output

The generated music file will be saved in the `output/` folder (created automatically).

---

## ⚡ Key Features

* LSTM-based sequence modeling for music generation
* Emotion-based dataset organization
* Temperature-based sampling for diverse outputs
* MIDI file generation for playback

---

## 🚀 Future Improvements

* Add real-time music generation UI
* Improve model with Transformer architecture
* Add audio (WAV/MP3) generation support
* Deploy as a web application

---

## 👩‍💻 Author

**Lahari Ponnaganti**

---
