# UrbanSound Classification with YAMNet and Deep Learning

This project provides an end-to-end pipeline for urban sound classification using pre-trained YAMNet features combined with a neural network model. It includes data loading, visualization, balancing, feature extraction, model training, evaluation, and deployment via a Flask web app.

---

## Table of Contents

- [Installation](#installation)
- [Libraries and Dependencies](#libraries-and-dependencies)
- [Data Preparation](#data-preparation)
- [Visualization](#visualization)
- [Data Balancing](#data-balancing)
- [Feature Extraction](#feature-extraction)
- [Model Building and Training](#model-building-and-training)
- [Evaluation](#evaluation)
- [Model Saving and Loading](#model-saving-and-loading)
- [Web Application Deployment](#web-application-deployment)
- [Usage](#usage)

---

## Installation

To run this project, first install the required packages:

```bash
pip install librosa numpy scipy matplotlib ipython tensorflow h5py seaborn pandas scikit-learn ipywidgets sounddevice wavio streamlit tensorflow-hub
```

---

## Libraries and Dependencies

The project uses the following core libraries:

- **Librosa**: Audio processing
- **NumPy & Pandas**: Data handling
- **Matplotlib & Seaborn**: Visualization
- **TensorFlow & Keras**: Deep learning model
- **TensorFlow Hub**: Pre-trained YAMNet model
- **scikit-learn**: Data splitting, oversampling, metrics
- **Sounddevice & Wavio**: Audio playback and recording (if needed)
- **Streamlit**: Optional UI (not used in the web app)
- **Flask**: Web app deployment

---

## Data Preparation

### Paths and Metadata Loading

Set your data directories and load the metadata:

```python
home_directory = os.path.expanduser("~")
base_directory = os.path.join(home_directory, 'Desktop', 'sounds', 'UrbanSound8k')
csv_path = os.path.join(base_directory, 'metadata', 'UrbanSound8K.csv')
audio_dir = os.path.join(base_directory, 'audio')

# Load metadata
meta_df = pd.read_csv(csv_path)
```

### Visualizations

- **Class Distribution**: Bar plot and pie chart showing number of samples per class.
- **Duration Histogram**: Distribution of sound event durations.
- **Correlation Matrix**: Relationships among numerical features.

### Data Balancing

To mitigate class imbalance, oversample minority classes:

```python
from sklearn.utils import resample

class_counts = meta_df['class'].value_counts()
max_samples = class_counts.max()

balanced_df = pd.DataFrame()
for cls in class_counts.index:
    class_subset = meta_df[meta_df['class'] == cls]
    oversampled = resample(class_subset, replace=True, n_samples=max_samples, random_state=42)
    balanced_df = pd.concat([balanced_df, oversampled])

balanced_df.to_csv(os.path.join(base_directory, 'balanced_metadata.csv'), index=False)
```

---

## Matching Audio Files with Metadata

Ensure each audio file has corresponding metadata, and filter audio files accordingly:

```python
# Rename for consistency
if 'slice_file_name' in balanced_df.columns:
    balanced_df.rename(columns={'slice_file_name': 'file_name'}, inplace=True)

audio_files = glob.glob(os.path.join(audio_dir, '**', '*.wav'), recursive=True)
audio_filenames = [os.path.basename(f) for f in audio_files]
metadata_files = balanced_df['file_name'].unique()

# Filter audio files
audio_files = [f for f in audio_files if os.path.basename(f) in metadata_files]
```

---

## Audio Visualization

Sample visualizations include waveform, spectrogram, and playback:

```python
def plot_waveform_and_spectrogram(file_path):
    y, sr = librosa.load(file_path, sr=None)
    plt.figure(figsize=(14, 6))
    # Waveform
    plt.subplot(1, 2, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    # Spectrogram
    plt.subplot(1, 2, 2)
    S = librosa.stft(y)
    S_DB = librosa.amplitude_to_db(np.abs(S), ref=np.max)
    librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='log', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.tight_layout()
    plt.show()
    display(Audio(y, rate=sr))
```

Visualize random samples:

```python
for file in random.sample(audio_files, 3):
    plot_waveform_and_spectrogram(file)
```

---

## Feature Extraction

Using YAMNet embeddings as features:

```python
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

def extract_yamnet_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    waveform = tf.convert_to_tensor(y, dtype=tf.float32)
    _, embeddings, _ = yamnet_model(waveform)
    return np.mean(embeddings, axis=0)
```

Prepare dataset:

```python
X, y = [], []
for _, row in balanced_df.iterrows():
    filepath = os.path.join(audio_dir, f'fold{row["fold"]}', row['file_name'])
    feature = extract_yamnet_features(filepath)
    if feature is not None:
        X.append(feature)
        y.append(row['class'])
X = np.array(X)
```

---

## Data Preparation for Model

Encode labels and split data:

```python
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

le = LabelEncoder()
y_encoded = le.fit_transform(y)
y_categorical = to_categorical(y_encoded)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_categorical, test_size=0.3, stratify=y_encoded, random_state=42
)
```

---

## Model Building and Training

Define a simple classifier:

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout

def build_model(input_shape, num_classes):
    model = Sequential()
    model.add(Dense(128, activation='relu', input_shape=input_shape))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))
    return model

model = build_model((X.shape[1],), len(le.classes_))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

Train with early stopping and checkpoint:

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)

history = model.fit(
    X_train, y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[early_stop, checkpoint]
)
```

---

## Model Evaluation

Assess performance:

```python
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy:.2f}")

# Plot accuracy & loss
plt.figure(figsize=(14,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.legend()

plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend()
plt.show()

# Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

y_pred = model.predict(X_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

cm = confusion_matrix(y_true, y_pred_classes)
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix')
plt.show()

print(classification_report(y_true, y_pred_classes, target_names=le.classes_))
```

---

## Saving and Loading the Model

Save the trained model and label encoder:

```python
import os
import pickle
import tensorflow as tf

os.makedirs('saved_models', exist_ok=True)
os.makedirs('saved_encoders', exist_ok=True)

model.save('saved_models/hybrid_yamnet.h5')
with open('saved_encoders/ylabel_e.pkl', 'wb') as f:
    pickle.dump(le, f)
```

Load and test saved model:

```python
from tensorflow.keras.models import load_model

loaded_model = load_model('saved_models/hybrid_yamnet.h5')
with open('saved_encoders/ylabel_e.pkl', 'rb') as f:
    loaded_le = pickle.load(f)

# Test prediction
sample_idx = 0
sample_feature = X_test[sample_idx].reshape(1, -1)
pred = loaded_model.predict(sample_feature)
pred_label = loaded_le.inverse_transform([np.argmax(pred, axis=1)[0]])[0]
print(f"Predicted label: {pred_label}")
```

---

## Web Application Deployment

### Flask App

The project includes a Flask app to upload audio, classify, and visualize results.

**app.py**:

```python
from flask import Flask, render_template, request, redirect, url_for
import os
from model_inference import classify_audio  # your inference functions

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/classify', methods=['POST'])
def classify():
    if 'audio_file' not in request.files:
        return redirect(url_for('home'))
    file = request.files['audio_file']
    if file.filename == '':
        return redirect(url_for('home'))
    save_path = os.path.join('static/recordings', file.filename)
    os.makedirs('static/recordings', exist_ok=True)
    file.save(save_path)
    class_name, dom_class, dom_pct, image_base64 = classify_audio(save_path)
    return render_template('results.html', class_name=class_name,
                           dominant_class=dom_class,
                           dominant_percentage=dom_pct,
                           image=image_base64)

if __name__ == '__main__':
    app.run(debug=True)
```

Ensure templates (`home.html`, `results.html`) are properly set up.

---

## Usage

1. Prepare your audio dataset and metadata.
2. Run data visualization and balancing scripts.
3. Extract features with YAMNet.
4. Train the classifier.
5. Save the model and encoder.
6. Launch the Flask app to classify new audio files via web interface.

---


## Note
- Adjust paths according to your environment.
- Consider GPU acceleration for faster training.
- Fine-tune model architecture and hyperparameters for improved accuracy.
- Extend the Flask app for more features as needed.

---

**Happy Sound Classifying!**
