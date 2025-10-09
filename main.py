from utils.extract_audio import extract_audio
from pipelines.translation import translate_text
from pipelines.transcription import transcribe_audio

def youtube_to_translation(youtube_url: str, target_lang: str = "ur"):
    """
    End-to-end pipeline:
    1. Extract audio from YouTube video.
    2. Transcribe it using Gemini.
    3. Translate transcript into target language.
    """

    print("\n🎬 Step 1: Extracting audio...")
    audio_path = extract_audio(youtube_url, output_folder="downloads")
    if not audio_path:
        print("❌ Failed to extract audio.")
        return None

    print("\n🗣️ Step 2: Transcribing audio...")
    transcript = transcribe_audio(audio_path)
    if not transcript or transcript.lower() == "audio is not clear.":
        print("❌ Failed to transcribe or unclear audio.")
        return None

    print("\n🌐 Step 3: Translating transcript...")
    translation = translate_text(transcript, target_lang=target_lang)

    print("\n✅ Done!\n")
    print("🎧 Transcript:\n", transcript)
    print("\n🈯 Translation:\n", translation)

    return {
        "transcript": transcript,
        "translation": translation,
        "audio_path": audio_path
    }


if __name__ == "__main__":
    youtube_url = "https://www.youtube.com/shorts/CIR6mZENrvE"
    result = youtube_to_translation(youtube_url, target_lang="ur")
