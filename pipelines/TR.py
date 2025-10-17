import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pipelines.transcription import transcribe_audio

test_file = r"D:\Programming\Python in Sublime\LLM Works\LangGraph Basic\Graphs\Youtube Rag Chatbot\downloaded_audio\Future is Here AI - Sundar Pichai.mp3"
print("Transcript:\n", transcribe_audio(test_file))