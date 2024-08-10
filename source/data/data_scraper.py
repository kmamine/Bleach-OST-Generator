import pandas as pd
import os
import re
from yt_dlp import YoutubeDL



playlists_ = [
    "https://www.youtube.com/playlist?list=OLAK5uy_kDV8usH6Ay6HbbQVshrW4-BKiORd8Fipc", 
    "https://www.youtube.com/playlist?list=OLAK5uy_nJlC_kCEkKS7Lkdeq9JLUGmLuY9fGoIX8", 
    "https://www.youtube.com/playlist?list=OLAK5uy_nbfdqEwOfK4bbRymFpOEn6QzqnG15Tb0Q",
    "https://www.youtube.com/playlist?list=OLAK5uy_lPx1XBYAMhENGe-ppwAPxNLyePCC-is3c"
]

output_folder = '/content/gdrive/MyDrive/Bleach-OST/soundtrack'



def download_and_extract_data(playlist_url, output_folder):
  ydl_opts = {
      'format': 'bestaudio/best',
      'outtmpl': os.path.join(output_folder, '%(title)s.%(ext)s'),
      'extract-audio': True,
      'audio-format': 'wav',
      'ignoreerrors': True,
      'quiet': True,
      'no-warnings': True
  }

  with YoutubeDL(ydl_opts) as ydl:
    info_dict = ydl.extract_info(playlist_url, download=True)
    entries = info_dict.get('entries', [])

    data = []
    for entry in entries:
      if entry:
        title = entry.get('title', '')
        likes = entry.get('like_count', None)
        comments = entry.get('comment_count', None)
        views = entry.get('view_count', None)
        upload_date = entry.get('upload_date', None)
        description = entry.get('description', '')
        duration = entry.get('duration', None)
        filename = f"{title}.wav"
        filepath = os.path.join(output_folder, filename)

        data.append({
            'title': title,
            'likes': likes,
            'comments': comments,
            'views': views,
            'release_date': upload_date,
            'description': description,
            'length': duration,
            'filepath': filepath
        })
      else:
        print(f"Skipping entry: {entry}")

  return pd.DataFrame(data)

output_folder = '/content/gdrive/MyDrive/Bleach-OST/soundtrack'
os.makedirs(output_folder, exist_ok=True)

all_data = []
for playlist_url in playlists_:
  df = download_and_extract_data(playlist_url, output_folder)
  all_data.append(df)

combined_df = pd.concat(all_data, ignore_index=True)
print(combined_df.head())


combined_df.to_csv(os.path.join(output_folder,'soundtrack_data.csv'), index=False)

