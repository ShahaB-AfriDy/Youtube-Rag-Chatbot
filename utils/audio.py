from yt_dlp import YoutubeDL
import os
import re

def extract_audio(youtube_url: str, output_folder: str = "downloaded_audio") -> str | None:
    match = re.search(r"shorts/([a-zA-Z0-9_-]{11})", youtube_url)
    if match:
        video_id = match.group(1)
        youtube_url = f"https://www.youtube.com/watch?v={video_id}"

    os.makedirs(output_folder, exist_ok=True)

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(output_folder, "%(title)s.%(ext)s"),
        "postprocessors": [{
            "key": "FFmpegExtractAudio",
            "preferredcodec": "mp3",
            "preferredquality": "192",
        }],
        "extractor_args": {"youtube": {"player_client": ["android"]}},
        "geo_bypass": True,
        "noplaylist": True,
        "quiet": False,
        "retries": 3,
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            filename = ydl.prepare_filename(info)
            audio_path = os.path.splitext(filename)[0] + ".mp3"
            # return os.path.abspath(audio_path)
            return rf"{os.path.abspath(audio_path)}"

    except Exception:
        return None


if __name__ == "__main__":
    audio_path = extract_audio(youtube_url="https://www.youtube.com/shorts/3s0dAdpZGx0")
    print(audio_path)