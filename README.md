# RNN Music Generation

This project implements an **Automatic Music Generation system** using  
**Recurrent Neural Networks (RNNs)** with **Long Short-Term Memory (LSTM)**.

The model learns musical patterns from **piano MIDI files** and generates
new music sequences automatically based on learned note relationships.

---

## Objective

- To generate music automatically using deep learning
- To learn sequential musical patterns such as pitch and rhythm
- To demonstrate the application of RNN–LSTM models in music generation

---

## Technologies Used

- Python 3.10
- TensorFlow / Keras
- music21
- NumPy
- MIDI (Musical Instrument Digital Interface)

---

## Project Structure


---

## How the Project Works

1. Piano MIDI files are collected as the dataset.
2. MIDI files are converted into numerical note sequences.
3. These sequences are used to train an LSTM-based RNN model.
4. The trained model predicts the next musical notes.
5. Predicted notes are converted back into a MIDI file.

---

## How to Run the Project

```bash
pip install -r requirements.txt
python src/data_preprocessing.py
python src/train_model.py
python src/generate_music.py
