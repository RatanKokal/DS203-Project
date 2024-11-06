import numpy as np
import pandas as pd
import librosa
import soundfile as sf

for file in range(1,116):
    if file < 10:
        filename = "0" + str(file) + "-MFCC.csv"
    else:
        filename = str(file) + "-MFCC.csv"
    mfcc_df = pd.read_csv(filename, header=None)
    sample_rate = 44100
    audio_signal = librosa.feature.inverse.mfcc_to_audio(mfcc_df.values, sr=sample_rate)
    sf.write(f'{file}.wav', audio_signal, sample_rate)

# # Load MFCCs from CSV
# mfcc_df = pd.read_csv('111-MFCC.csv', header=None)  # Adjust if your CSV has headers
# # mfccs = mfcc_df.values.T  # Transpose if necessary

# # Assuming you have the original sample rate
# sample_rate = 44100  # Replace with your actual sample rate

# # Reconstruct the audio signal from MFCCs
# audio_signal = librosa.feature.inverse.mfcc_to_audio(mfcc_df.values, sr=sample_rate)

# # Save the audio signal to a WAV file
# sf.write('output.wav', audio_signal, sample_rate)
