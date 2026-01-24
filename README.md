# RNN Music Generation

This project implements an Automatic Music Generation system using
Recurrent Neural Networks (RNNs) with Long Short-Term Memory (LSTM).

The model learns musical patterns from piano MIDI files and generates
new music sequences automatically.

---

## Project Structure

RNN_MusicGeneration/
│
├── data/
│   └── midi/                 # Input MIDI dataset
│
├── src/
│   ├── data_preprocessing.py # Converts MIDI to note sequences
│   ├── model.py              # LSTM model architecture
│   ├── train_model.py        # Model training script
│   └── generate_music.py     # Music generation script
│
├── output/
│   └── generated_music.mid   # Generated music output
│
├── best_model.h5             # Trained model weights
├── requirements.txt          # Required Python libraries
└── README.md
