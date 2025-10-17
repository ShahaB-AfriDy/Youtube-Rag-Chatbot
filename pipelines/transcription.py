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
    model="gemini-2.5-flash",
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
                "Please transcribe the following audio into text.\n"
                "Guidelines:\n"
                "1. Write down exactly what is spoken as accurately as possible.\n"
                "2. Avoid summarizing or changing the meaning.\n"
                "3. Keep the original wording and tone.\n"
                "4. Use natural punctuation for readability (commas, periods, etc.).\n"
                "5. Do not include any extra notes, labels, or introductions.\n"
                "6. If the audio is mostly unclear or unintelligible, respond with: 'Audio is not clear.'\n"
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
    # test_file = r"D:\MindScribe\backends\downloads\WHILE vs DURING - English Grammar Difference Explained ｜ Learn English with Ananya.mp3"
    # test_file = r"D:\Programming\Python in Sublime\LLM Works\LangGraph Basic\Graphs\Youtube Rag Chatbot\downloaded_audio\A.I. Teaches Programming.mp3"
    # test_file = r"D:\Programming\Python in Sublime\LLM Works\LangGraph Basic\Graphs\Youtube Rag Chatbot\downloaded_audio\OpenAI's ChatGPT creates an operating system.mp3"
    test_file = r" D:\Programming\Python in Sublime\LLM Works\LangGraph Basic\Graphs\Youtube Rag Chatbot\downloaded_audio\A.I. Teaches Programming.mp3"
    print("Transcript:\n", transcribe_audio(test_file))