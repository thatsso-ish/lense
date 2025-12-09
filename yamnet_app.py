import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import streamlit as st
import tensorflow as tf
import tensorflow_hub as hub
import pickle
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

# Get the base directory of the script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load the pre-trained CNN model and label encoder
model_path = os.path.join(BASE_DIR, 'saved_models', 'hybrid_yamnet.h5')
loaded_cnn_model = tf.keras.models.load_model(model_path)

encoder_path = os.path.join(BASE_DIR, 'saved_encoders', 'ylabel_e.pkl')
with open(encoder_path, 'rb') as le_file:
    loaded_le = pickle.load(le_file)

# Load YAMNet model
yamnet_model = hub.load('https://tfhub.dev/google/yamnet/1')

# Load class mapping
class_mapping = {
    0: 'air_conditioner',
    1: 'car_horn',
    2: 'children_playing',
    3: 'dog_bark',
    4: 'drilling',
    5: 'engine_idling',
    6: 'gun_shot',
    7: 'jackhammer',
    8: 'siren',
    9: 'street_music',
    10: 'neutral'
}

# Function to extract features using YAMNet
def extract_yamnet_features(file_path):
    y, sr = librosa.load(file_path, sr=16000)
    waveform = tf.convert_to_tensor(y, dtype=tf.float32)
    _, yamnet_embeddings, _ = yamnet_model(waveform)
    embedding = np.mean(yamnet_embeddings, axis=0)  # Average pooling the embeddings
    return embedding

# Function to predict sound class and get distribution
def predict_sound(file_path):
    feature = extract_yamnet_features(file_path)
    feature = feature.reshape(1, -1)
    pred = loaded_cnn_model.predict(feature)
    pred_class = np.argmax(pred)
    confidence = np.max(pred) * 100  # Convert confidence to percentage
    percentages = (pred[0] * 100).round(2)  # Convert all class probabilities to percentages
    return pred_class, confidence, percentages

# Function to plot waveform and mel spectrogram
def plot_waveform_and_spectrogram(y, sr):
    plt.figure(figsize=(14, 6))
    # Plot Waveform
    plt.subplot(1, 2, 1)
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')

    # Plot Mel Spectrogram
    plt.subplot(1, 2, 2)
    S = librosa.feature.melspectrogram(y=y, sr=sr)
    S_DB = librosa.power_to_db(S, ref=np.max)
    librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel', cmap='magma')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel Spectrogram')

    plt.tight_layout()
    st.pyplot(plt)

# Streamlit Application
st.title("Sound Classifier App")

# Upload Audio
uploaded_file = st.file_uploader("Upload a .wav or .mp3 file", type=["wav", "mp3"])
if uploaded_file is not None:
    # Save the uploaded file temporarily
    file_path = "temp.wav"  # Default to .wav
    if uploaded_file.name.endswith('.mp3'):
        file_path = "temp.mp3"
    
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())

    # Audio playback
    st.audio(file_path, format="audio/wav")

    # Predict sound class
    pred_class, confidence, percentages = predict_sound(file_path)
    st.write(f"Predicted Class: {class_mapping[pred_class]} with confidence {confidence:.2f}%")

    # Load and plot waveform and mel-spectrogram
    y, sr = librosa.load(file_path, sr=16000)
    plot_waveform_and_spectrogram(y, sr)

    # Display percentage distribution of all classes (excluding 'neutral')
    st.subheader("Class Confidence Percentages (Excluding Neutral)")
    percentage_data = {
        class_mapping[i]: percentages[i] 
        for i in range(len(class_mapping)) 
        if class_mapping[i] != "neutral"
    }
    percentage_df = pd.DataFrame(percentage_data.items(), columns=["Class", "Confidence (%)"])
    st.table(percentage_df)
