import os
import logging
import streamlit as st
import numpy as np
import librosa, librosa.display
import matplotlib.pyplot as plt
from PIL import Image
from settings import IMAGE_DIR, DURATION, WAVE_OUTPUT_FILE
from src.sound import sound
from src.model import CNN
from setup_logging import setup_logging

setup_logging()
logger = logging.getLogger('app')

def init_model():
    cnn = CNN((128, 87))
    cnn.load_model()
    return cnn

def get_spectrogram(type='mel'):
    logger.info("Extracting spectrogram")
    y, sr = librosa.load(WAVE_OUTPUT_FILE, duration=DURATION)
    ps = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    logger.info("Spectrogram Extracted")
    format = '%+2.0f'
    if type == 'DB':
        ps = librosa.power_to_db(ps, ref=np.max)
        format = ''.join([format, 'DB'])
        logger.info("Converted to DB scale")
    return ps, format

def display(spectrogram, format):
    plt.figure(figsize=(10, 4))
    librosa.display.specshow(spectrogram, y_axis='mel', x_axis='time')
    plt.title('Mel-frequency spectrogram')
    plt.colorbar(format=format)
    plt.tight_layout()
    st.pyplot(clear_figure=False)

def main():
    st.set_page_config(page_title="Guitar Chord Recognition", layout="wide")
    
    st.sidebar.title('Navigation')
    option = st.sidebar.radio('Select an option', ['Home', 'Record', 'Play', 'Classify', 'Display Spectrogram'])
    
    st.title("Music Chord Recognition")
    
    if option == 'Home':
        st.header("Welcome to the Guitar Chord Recognition App")
        image = Image.open(os.path.join(IMAGE_DIR, 'immmg.jpg'))
        st.image(image, use_column_width=True)
        st.markdown("""
            This application helps you recognize guitar chords from audio recordings.
            Use the navigation on the left to record, play, classify chords, or view the spectrogram of your recording.
        """)
    
    elif option == 'Record':
        st.header("Record Your Chord")
        if st.button('Start Recording'):
            with st.spinner(f'Recording for {DURATION} seconds ....'):
                sound.record()
            st.success("Recording completed")
    
    elif option == 'Play':
        st.header("Play Your Recorded Audio")
        if st.button('Play Recorded Audio'):
            try:
                audio_file = open(WAVE_OUTPUT_FILE, 'rb')
                audio_bytes = audio_file.read()
                st.audio(audio_bytes, format='audio/wav')
            except:
                st.error("Please record sound first")
    
    elif option == 'Classify':
        st.header("Classify the Recorded Chord")
        if st.button('Classify Chord'):
            cnn = init_model()
            with st.spinner("Classifying the chord"):
                chord = cnn.predict(WAVE_OUTPUT_FILE, False)
            st.success("Classification completed")
            st.write(f"### The recorded chord is **{chord}**")
            if chord == 'N/A':
                st.warning("Please record sound first")
    
    elif option == 'Display Spectrogram':
        st.header("View Spectrogram")
        if os.path.exists(WAVE_OUTPUT_FILE):
            spectrogram, format = get_spectrogram(type='mel')
            display(spectrogram, format)
        else:
            st.error("Please record sound first")

if __name__ == '__main__':
    main()
