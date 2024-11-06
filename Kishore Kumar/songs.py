from pytubefix import YouTube
from pytubefix.cli import on_progress
import os
from youtube_search import YoutubeSearch
from moviepy.editor import VideoFileClip

# Create a list to hold YouTube links
links = []

# Read video queries from the text file
with open('kk.txt', "r") as file:
    for line in file:
        query = line.strip() + " Kishore Kumar"  # Add artist name for a more specific search
        results = YoutubeSearch(query, max_results=1).to_dict()
        for result in results:
            video_id = result["id"]
            link = f"https://www.youtube.com/watch?v={video_id}"
            links.append(link)

# Download audio for each link
for link in links:
    try:
        yt = YouTube(link, on_progress_callback=on_progress)
        print(f"Downloading: {yt.title}")

        # Download audio as MP4
        ys = yt.streams.filter(file_extension='mp4').get_highest_resolution()
        mp4_file_path = ys.download(output_path='.', filename=f"{yt.title}.mp4")

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
