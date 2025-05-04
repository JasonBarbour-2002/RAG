import os
import yt_dlp

url = "https://www.youtube.com/watch?v=dARr3lGKwk8"

os.makedirs("Data", exist_ok=True)  # Create the Data directory if it doesn't exist
ydl_opts = {
    'format': 'bestvideo+bestaudio/best',
    'outtmpl': 'Data/%(title)s.%(ext)s',  # Save as the video title
}

with yt_dlp.YoutubeDL(ydl_opts) as ydl:
    ydl.download([url])