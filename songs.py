# from pytubefix import YouTube
# from pytubefix.cli import on_progress
 
# url = "https://www.youtube.com/watch?v=RE87rQkXdNw"
 
# yt = YouTube(url, on_progress_callback = on_progress)
# print(yt.title)
 
# ys = yt.streams.get_audio_only()
# ys.download(mp3=True)

from pytubefix import YouTube
from pytubefix.cli import on_progress
import os
from youtube_search import YoutubeSearch

l = []
with open('asha_bhosale.txt', "r") as file:
        for line in file:
                query = line + "Asha Bhosale"
                results = YoutubeSearch(query, max_results=1).to_dict()
                for result in results:
                        video_id = result["id"]
                        link = f"https://www.youtube.com/watch?v={video_id}"
                        l.append(link)

for line in l:
        yt = YouTube(line, on_progress_callback = on_progress)
        print(yt.title)
 
        ys = yt.streams.get_audio_only()
        ys.download(mp3=True)
        
