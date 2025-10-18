from yt_dlp import YoutubeDL
import os
import re

def extract_audio(youtube_url: str, output_folder: str = "downloaded_audio") -> tuple[str | None, str | None]:
    """
    Downloads YouTube video audio as MP3 and returns:
    (absolute file path, video title)
    """

    # Handle YouTube Shorts links
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
            abs_path = os.path.abspath(audio_path)
            title = info.get("title", None)
            return abs_path, title

    except Exception as e:
        print(f"❌ Error: {e}")
        return None, None


if __name__ == "__main__":
    path, title = extract_audio("https://www.youtube.com/shorts/3s0dAdpZGx0")
    print("Audio File Path:", path)
    print("Video Title:", title)
