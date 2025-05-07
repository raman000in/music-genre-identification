# 🎵 Music Genre Identification using Deep Learning

This project uses deep learning to classify music audio files into genres like rock, pop, jazz, classical, and more.

## 📦 Dataset

- Used GTZAN Dataset (10 genres, `.au` files format).
- Extracted features: MFCC (Mel Frequency Cepstral Coefficients)

## 📊 Model

- Deep Learning Model → Dense layers based Neural Network (using Tensorflow / Keras).
- Input → MFCC (mean across time → shape (40,))
- Output → 10 genre classes.

## 📁 Files

- `model/music_genre_model.keras` → Saved trained model
- `notebooks/music_genre_training.ipynb` → Full code including:
  - Audio loading
  - MFCC feature extraction
  - Model building and training
  - Saving model
  - Making predictions
- `test_predict.py` → How to predict new audio files (optional, if you add)

## 🚀 How to use

### Load model and predict a song

```python
from keras.models import load_model
import librosa
import numpy as np

model = load_model("model/music_genre_model.keras")

# Load new audio file
y, sr = librosa.load("genres/blues/blues.00005.au", duration=30)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
mfcc = np.mean(mfcc.T, axis=0)
mfcc = mfcc.reshape(1, 40)

# Predict
prediction = model.predict(mfcc)
predicted_class = np.argmax(prediction)

genre_map = {0: 'blues', 1: 'classical', 2: 'country', 3: 'disco', 4: 'hiphop', 
             5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'}

print("Predicted Genre:", genre_map[predicted_class])
