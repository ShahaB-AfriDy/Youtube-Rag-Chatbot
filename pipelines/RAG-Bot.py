import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from typing import List, Annotated
from pydantic import BaseModel
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph,add_messages,END

from langgraph.store.postgres import PostgresStore
from langgraph.store.base import BaseStore

from utils.db import store, checkpointer,close_connections

from utils.audio import extract_audio
from pipelines.transcription import transcribe_audio

from dotenv import load_dotenv
import os

load_dotenv()

# ---------------------- MODEL SETUP ----------------------
model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "{system_message}"),
    MessagesPlaceholder("messages")
])

llm_model = prompt_template | model


# ---------------------- RAG STATE ----------------------
from typing import List, Annotated,Optional
from pydantic import BaseModel, Field
from langchain.schema import Document
from langchain_core.messages import AnyMessage


from urllib.parse import urlparse, parse_qs

def get_youtube_video_id(url: str) -> str:
    """
    Extracts the YouTube video ID from a URL.
    Supports:
      - https://www.youtube.com/watch?v=VIDEO_ID
      - https://youtu.be/VIDEO_ID
      - https://www.youtube.com/shorts/VIDEO_ID
    """
    if not url:
        return None

    if "youtu.be" in url:
        return url.split("/")[-1]
    elif "youtube.com" in url:
        parsed = urlparse(url)
        if parsed.path.startswith("/shorts/"):
            return parsed.path.split("/")[2]  # get ID after /shorts/
        query = parse_qs(parsed.query)
        return query.get("v", [None])[0]
    return None


class RAGState(BaseModel):
    """
    Represents the state of a Retrieval-Augmented Generation (RAG) conversation.
    """
    messages: Annotated[List[AnyMessage], add_messages]
    youtube_link: Annotated[Optional[str], Field(default=None, description="YouTube video URL (if provided)")]
    transcription: Annotated[Optional[str], Field(default=None, description="Full transcription of the video")]
    retrieved_docs: Annotated[List[Document], Field(default_factory=list, description="Retrieved document chunks for context")]
    final_answer: Annotated[Optional[str], Field(default=None, description="Final generated response from the LLM")]

# ---------------------- GRAPH NODES ----------------------

def Check_URL_Node(state: RAGState, store: BaseStore,config: RunnableConfig):
    
    """
    Checks if this user has already transcribed this YouTube video.
    Returns one of: "no_url", "transcribe", "not_transcribe"
    """
    user_id = config["configurable"]["user_id"]
    # Check URL presence
    if not state.youtube_link or not state.youtube_link.strip():
        print("if No URL".center(100,"*"))
        return "no_url"

    # Check if transcription already exists
    namespace = (user_id, get_youtube_video_id(state.youtube_link))
    try:
        results = store.search(namespace,filter={"url":state.youtube_link},limit=1)
    except Exception as e:
        print(f"DB search error: {e}")
        results = []
    print(state.youtube_link)
    if results:
        print("Yes URL EXIST IN DB".center(100,"-"))
        return "not_transcribe"
    else:
        print("No URL IS NOT EXIST IN DB".center(100,"-"))
        return "transcribe"


def No_URL_Node(state: RAGState):
    print("No url node".center(100,"*"))
    """Handles case where user didn't provide a YouTube URL."""
    state.final_answer = "Please paste a YouTube video link first before asking questions."
    return state


def Transcribe_Node(state: RAGState, store: BaseStore, config: RunnableConfig):
    print("Transcribe".center(100,"*"))
    """Downloads, transcribes, and stores embeddings."""
    user_id = config["configurable"]["user_id"]

    video_id = get_youtube_video_id(state.youtube_link)
    namespace = (user_id, video_id)
    # Step 1: Download audio
    audio_path = extract_audio(state.youtube_link, output_folder="downloaded_audio")
    print(F"Audio Path: {audio_path}")
    # Step 2: Transcribe
    
    state.transcription = transcribe_audio(audio_path)
    print(F"Transcriptions->>:  {state.transcription}")

    # Step 3: Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(state.transcription)

    # Step 4: Store each chunk as embedding
    for i, chunk in enumerate(chunks):
        key = f"chunk_{i+1}"
        metadata = {"text": chunk, "url": state.youtube_link}
        store.put(namespace, key, metadata, index=["text"])
    print("Successfully data put in the db")
    return state


def Retrieval_Node(state: RAGState, store: BaseStore, config: RunnableConfig):
    print("Retrieval".center(100,"*"))
    user_id = config["configurable"]["user_id"]
    namespace = (user_id, get_youtube_video_id(state.youtube_link))

    if not state.messages or not state.messages[-1].content.strip():
        print("No user question provided — skipping retrieval for now.")
        state.retrieved_docs = []
        return state

    query = state.messages[-1].content
    result = store.search(namespace, query=query, limit=3)

    # Wrap text chunks in Document objects
    retrieved = [
        Document(page_content=r.value["text"], metadata={"url": state.youtube_link})
        for r in result
    ] if result else []

    state.retrieved_docs = retrieved
    print(f"Retrieved {len(retrieved)} relevant chunks for query: {query}")
    return state



def Chat_Node(state: RAGState, store: BaseStore, config: RunnableConfig):
    print("Chat_Node".center(100,"*"))
    """
    Generates the final answer using the retrieved context.
    """
    if not state.messages or not state.messages[-1].content.strip():
        state.final_answer = (
            "The video has been successfully transcribed and stored.\n"
            "You can now ask me questions about its content!"
        )
        return state

    context = "\n".join(doc.page_content for doc in state.retrieved_docs) if state.retrieved_docs else "No relevant context found."

    system_message = SystemMessage(content=f"You are a helpful tutor. Use this context:\n{context}")

    response = llm_model.invoke({
        "system_message": system_message,
        "messages": [{"role": "user", "content": state.messages[-1].content}],
    })

    state.final_answer = response.content
    return state


# ---------------------- GRAPH BUILD ----------------------
graph_builder = StateGraph(RAGState)


graph_builder.add_node(node="Check_URL_Node", action= lambda state:state)
# graph_builder.add_node(node="No_URL_Node", action= lambda state:state)    

# graph_builder.add_node("Check_URL_Node", Check_URL_Node)
graph_builder.add_node("No_URL_Node", No_URL_Node)
        # <-- move here (before it's referenced)
graph_builder.add_node("Transcribe_Node", Transcribe_Node)
graph_builder.add_node("Retrieval_Node", Retrieval_Node)
graph_builder.add_node("Chat_Node", Chat_Node)


graph_builder.set_entry_point("Check_URL_Node")


graph_builder.add_conditional_edges(
    source="Check_URL_Node",
    path=Check_URL_Node,
    # path=lambda state, store, config, result: result,
    path_map={
        "no_url": "No_URL_Node",
        "transcribe": "Transcribe_Node",
        "not_transcribe": "Retrieval_Node"
    }
)

graph_builder.add_edge("Transcribe_Node", "Retrieval_Node") 
graph_builder.add_edge("Retrieval_Node", "Chat_Node") 
graph_builder.add_edge("No_URL_Node",END)
graph_builder.set_finish_point("Chat_Node")


# app = graph_builder.compile(checkpointer=checkpointer,store=store)
app = graph_builder.compile(store=store)

# URL = "https://www.youtube.com/shorts/6wHscF7GE6A"
# URL = "https://www.youtube.com/shorts/x2VefKXyLko"
# URL = "https://www.youtube.com/shorts/x2VefKXyLko"
# URL = "https://www.youtube.com/shorts/FH1AMAKgdn4"
# URL = "https://www.youtube.com/watch?v=wjZofJX0v4M"
# URL = "https://www.youtube.com/shorts/je4Q1vBCpok"
URL = "https://www.youtube.com/shorts/XJ1yWRwZ6JQ"

# state = RAGState(
#     messages=[HumanMessage(content="What telling the Presenter in Video!")],
#     youtube_link=URL,
#     retrieved_docs=[]  # MUST be empty list, not containing any strings
# )
# msg = "Context Size: The network can only process a fixed number of vectors at a time"
# "Vector Association: Each token is associated with a vector (a list of numbers) meant to encode its meaning"

msg = "Hi i need help in maths"
state = RAGState(
    messages=[HumanMessage(content=msg)],
    youtube_link=URL,
    retrieved_docs=[]  # MUST be empty list, not containing any strings
)

config = {"configurable": {
        "user_id": "user_001",
        "thread_id": "1",
}}

result = app.invoke(state, config=config)
print(result["final_answer"])





# close_connections()

# "https://www.youtube.com/shorts/5ZWub9UEJiE"
