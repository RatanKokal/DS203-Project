import audioread

file_name = 'Michael Jackson - Another Part of Me (Official Video).mp3'
try:
    with audioread.audio_open(file_name) as audio_file:
        sr = audio_file.samplerate
        print(f"Sampling Rate: {sr} Hz")
except Exception as e:
    print(f"Error opening audio file: {str(e)}")
