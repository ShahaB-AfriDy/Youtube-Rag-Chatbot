
from dotenv import load_dotenv
load_dotenv()

import sys
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    api_key=os.getenv("GOOGLE_API_KEY"),
)

def translate_text(text: str, target_lang: str = "en") -> str:
    """
    Translate given text into target language (default English).
    Returns only the translated text, with no extra explanations or paraphrasing.
    """

    if not text.strip():
        raise ValueError("Input text cannot be empty.")

    if target_lang.lower() == "en":
        return text  # no need to translate

    # Prompt: force strict, literal translation
    message = HumanMessage(
    content=[
        {
            "type": "text",
            "text": (
                f"Your task is to ALWAYS detect the source language of the provided text and then translate it into the target language with code: {target_lang}.\n"
                f"⚠️ STRICT RULES:\n"
                f"1. Always output the translation, even if the source text looks similar to the target language.\n"
                f"2. Output only the translated text — no labels, explanations, or notes.\n"
                f"3. Keep the meaning fully accurate — do not paraphrase or summarize.\n"
                f"4. Preserve the tone, style, and formatting as much as possible.\n"
                f"5. If the text is unreadable or garbled, respond only with: 'Text is not clear.'\n\n"
                f"TEXT TO TRANSLATE:\n{text}"
            ),
        }
    ]
)



    translation = llm.invoke([message]).content.strip()
    return translation



if __name__ == "__main__":
    sample_text = " While versus during, do these two words confuse you? Come on, let's understand. While is a conjunction and it's used with a subject plus verb. It is used to show that two actions happen at the same time. Example, I was cooking while she was watching TV. She smiled while talking to me. Now, during, it's a preposition and we use it with a noun, not with a full sentence. It's used to show something happened in a period of time, specific period of time. I met many people during the conference. Now, a conference happens in a specific period of time. I fell asleep during the movie. Now, these two sentences, where will you use while and during? Well, let me know in the comments."
    # sample_text = "As soon as you place thi/s shield on the ground and step on it, your shoe gets covered immediately..."
    print("Translation:\n", translate_text(sample_text, target_lang="ur"))
