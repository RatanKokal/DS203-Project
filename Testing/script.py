import numpy as np
import pandas as pd
import librosa
import soundfile as sf

for file in range(2,3):
    filename = str(file) + "_MFCC.csv"
    mfcc_df = pd.read_csv(filename, header=None)
    sample_rate = 44100
    audio_signal = librosa.feature.inverse.mfcc_to_audio(mfcc_df.values, sr=sample_rate)
    sf.write(f'{file}_updated.mp3', audio_signal, sample_rate)