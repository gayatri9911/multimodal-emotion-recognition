import librosa
import numpy as np

def extract_features(file_path):

    try:

        audio, sample_rate = librosa.load(
            file_path,
            duration=3,
            offset=0.5
        )

        mfcc = librosa.feature.mfcc(

            y=audio,

            sr=sample_rate,

            n_mfcc=40
        )

        mfcc = mfcc.T

        return mfcc

    except Exception as e:

        print("Error processing:", file_path)

        return None