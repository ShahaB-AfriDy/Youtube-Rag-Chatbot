from yt_dlp import YoutubeDL
import os
import re

def extract_audio(youtube_url, output_folder="."):
    """
    Extract audio from a YouTube video (including Shorts) and save as MP3.
    Handles Shorts URLs, Android client workaround, and safe filename output.
    """
    # ✅ Convert Shorts URL → Watch URL
    match = re.search(r"shorts/([a-zA-Z0-9_-]{11})", youtube_url)
    if match:
        video_id = match.group(1)
        youtube_url = f"https://www.youtube.com/watch?v={video_id}"
        print(f"🔁 Converted Shorts URL to: {youtube_url}")

    # ✅ Ensure output directory exists
    os.makedirs(output_folder, exist_ok=True)

    # ✅ yt-dlp options for stable audio download
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
        "retries": 3,  # Retry in case of transient errors
    }

    try:
        print("🔍 Downloading audio... please wait...")
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            filename = ydl.prepare_filename(info)
            audio_path = os.path.splitext(filename)[0] + ".mp3"

        print(f"\n✅ Audio saved successfully: {audio_path}")
        return audio_path

    except Exception as e:
        print(f"\n❌ Error extracting audio: {e}")
        return None


if __name__ == "__main__":
    youtube_link = "https://www.youtube.com/shorts/CIR6mZENrvE"
    extract_audio(youtube_link, output_folder="downloaded_audio")
