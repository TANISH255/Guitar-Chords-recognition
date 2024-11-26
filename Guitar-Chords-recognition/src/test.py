import logging
import numpy as np
import librosa
from src.model import CNN
from settings import MODEL_JSON, MODEL_H5

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('test_model')

def test_model(filepath):
    # Initialize the CNN model
    cnn = CNN((128, 87))  # Ensure this matches the input shape during training

    # Load the model
    logger.info("Loading model")
    cnn.load_model()

    # Predict the chord from the audio file
    logger.info("Starting prediction")
    chord = cnn.predict(filepath, loadmodel=False)

    # Output the result
    if chord is not None:
        logger.info(f"The predicted chord is: {chord}")
    else:
        logger.error("Prediction failed, returned None")

if __name__ == '__main__':
    # Test with a specific audio file
    test_audio_file = 'recorded.wav'
    test_model(test_audio_file)
