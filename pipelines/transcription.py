import sys
import os
from dotenv import load_dotenv
load_dotenv()
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# from config import GEMINI_API_KEY
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    api_key=os.getenv("GOOGLE_API_KEY"),
)

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribe audio file into text using Gemini.
    Returns ONLY the spoken words faithfully (no summarization, no rephrasing).
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    with open(audio_path, "rb") as f:
        audio_bytes = f.read()

    message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": (
                "Transcribe the following audio into text.\n"
                "⚠️ STRICT RULES:\n"
                "1. Output only the exact spoken words, as heard.\n"
                "2. Do NOT summarize, rephrase, or interpret.\n"
                "3. Preserve the original meaning and word choice.\n"
                "4. Use only minimal punctuation where natural (commas, periods).\n"
                "5. Do NOT add labels, explanations, or prefixes (e.g., 'Transcript:').\n"
                "6. If the audio is unclear, noisy, or unintelligible, respond only with: 'Audio is not clear.'\n"
            ),
        },
        {"type": "media", "mime_type": "audio/mpeg", "data": audio_bytes},
    ]
)



    try:
        transcript = llm.invoke([message]).content.strip()
    except Exception as e:
        raise RuntimeError(f"Gemini transcription failed: {e}")

    return transcript


if __name__ == "__main__":
    test_file = r"D:\MindScribe\backends\downloads\WHILE vs DURING - English Grammar Difference Explained ｜ Learn English with Ananya.mp3"
    print("Transcript:\n", transcribe_audio(test_file))