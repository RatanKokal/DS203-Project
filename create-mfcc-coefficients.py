# Uncomment the following line if you do not have the Python module 'librosa' installed
# !pip install librosa

import os
import numpy as np
import pandas as pd
import librosa
from matplotlib import pyplot as plt

# Function to create MFCC coefficients given an audio file

def create_MFCC_coefficients(file_name):

    sr_value = 44100
    n_mfcc_count = 20
    
    try:
        # Load the audio file using librosa
        y, sr = librosa.load(file_name, sr=sr_value)
              
        # Compute MFCC coefficients for the segment
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc_count)
        #print(f"file_name: {file_name}: y={len(y)}, sr={sr}, mfccs matrix:{np.shape(mfccs)}")

        librosa.display.specshow(mfccs, x_axis='time', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.title('MFCC')
        plt.xlabel('Time')
        plt.ylabel('MFCC Coefficient')
        plt.show()
        
        # Create and return MFCC dataframe
        coeff_df = pd.DataFrame(mfccs)
        
        return coeff_df

    except Exception as e:
       print(f"Error creating MFCC coefficients: {file_name}:{str(e)}")

df = create_MFCC_coefficients('Piya Tu Ab To Aaja 4K - Asha Bhosle, R D Burman - Helen, Jeetendra, Asha Parekh - Caravan 1971 Songs - SuperHit Gaane (youtube).mp3')
df.to_csv('ab1-MFCC.csv', index=False, header=False)