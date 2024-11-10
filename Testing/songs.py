from pytubefix import YouTube
from pytubefix.cli import on_progress
import os
from youtube_search import YoutubeSearch
from moviepy.editor import VideoFileClip

# Create a list to hold YouTube links
links = ['RqQ4asI70Bg', 'q2hyS7XmOyw', 'kIdQPol-Ejg', '39LvqCNokz4', 'rFZ8jTj66xc', 'uTQ4sJMTLzc', '154OCvW6PQU', 'vXV0tIf3p3Y', 'Vqag_PZRLtk', 'X0rRnUQHqSY', 'HUjr_XyOJeE', 'IPIcaPgSZsA', '34oGNAwiDUg', 'e41Bwihtke8', 'LcEr4XwXMAg', 'fBvYwF6NMnY', 'A0nmixFJ0Hc', 'q7c6zqn4L8g', 'Bt1e4JKY-TM', 'A0nmixFJ0Hc', 'PUMD8dxaABs', 'fBvYwF6NMnY', '6XZvyAkHwA8', 'LcEr4XwXMAg', '9Sxe2tzbBiU', 'NPRxBH6Fh_M', '2L4vb7kdRjA', 'Xc0xRyHLiAo', 'F0GYEj_jhWY', '2OOcsAnl9LM', 'MQ9iypvxOgc', 'ZmBiB04dhSw', '53zSx8rq5t0', 'XyubcAXY56Q', 'vPKp29Luryc', 'PUMD8dxaABs', '38UUz15fEzo']

# Download audio for each link
for link in links:
    try:
        yt = YouTube("https://www.youtube.com/watch?v=" + link, on_progress_callback=on_progress)
        print(f"Downloading: {yt.title}")

        # Download audio as MP4
        ys = yt.streams.filter(file_extension='mp4').get_highest_resolution()
        mp4_file_path = ys.download(output_path='Testing', filename=f"{yt.title}.mp4")

        # Convert MP4 to MP3
        print(f"Converting {mp4_file_path} to MP3...")
        clip = VideoFileClip(mp4_file_path)
        ext_audio = clip.audio

        # Prepare the MP3 filename
        mp3_file_path = os.path.splitext(mp4_file_path)[0] + '.mp3'

        # Write the audio file as MP3
        ext_audio.write_audiofile(mp3_file_path)

        # Close the audio and video clips
        ext_audio.close()
        clip.close()

        # Delete the original MP4 file to save space
        os.remove(mp4_file_path)
        print(f"Converted and deleted {mp4_file_path}. MP3 saved as {mp3_file_path}.")

    except Exception as e:
        print(f"Error processing {link}: {str(e)}")

print("All downloads and conversions completed.")
