import librosa
import librosa.display
import numpy as np
import soundfile as sf

# Load an initial audio file (you can use any .wav or .flac file you have)
audio_file = '2.wav'
y, sr = librosa.load(audio_file, sr=44100)  # Load with sample rate 44100 Hz

# Compute the initial MFCCs (n_mfcc = 20)
mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

# Repeat the process 50 times
for i in range(12):
    # Inverse transform: MFCC to audio
    audio_reconstructed = librosa.feature.inverse.mfcc_to_audio(mfcc, sr=sr)
    
    # Compute MFCCs again for the reconstructed audio
    mfcc = librosa.feature.mfcc(y=audio_reconstructed, sr=sr, n_mfcc=20)
    
    # Optional: Save the intermediate audio (e.g., every 10 iterations)
    if i % 10 == 0:
        sf.write(f'output_audio_iteration_{i}.wav', audio_reconstructed, sr)

# Save the final audio after 50 iterations
sf.write('final_output_audio.wav', audio_reconstructed, sr)

print("Process completed and final audio saved.")
