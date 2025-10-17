from typing import List, Annotated
from pydantic import BaseModel
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.vectorstores.pgvector import PGVector
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.messages import AnyMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph
from langgraph.prebuilt import add_messages
from langgraph.store.postgres import PostgresStore
from langgraph.store.base import BaseStore

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
class RAGState(BaseModel):
    messages: Annotated[List[AnyMessage], add_messages]
    youtube_link: str
    transcription: str = ""
    retrieved_docs: List[Document] = []
    summarized_context: str = ""
    final_answer: str = ""


# ---------------------- GRAPH NODES ----------------------

def Check_URL_Node(state: RAGState, store: BaseStore, config: RunnableConfig):
    """
    Checks if this user has already transcribed this YouTube video.
    If yes -> skip transcription.
    If no  -> move to transcription.
    """
    user_id = config["configurable"]["user_id"]
    namespace = (user_id, state.youtube_link)

    results = store.search(namespace, query="*", limit=1)

    if results:  # video already exists for this user
        return "not_transcribe"
    else:
        return "transcribe"


def Transcribe_Node(state: RAGState, store: BaseStore, config: RunnableConfig):
    """Downloads, transcribes, and stores embeddings."""
    user_id = config["configurable"]["user_id"]
    namespace = (user_id, state.youtube_link)

    # Step 1: Download audio
    audio_path = extract_audio(state.youtube_link, output_folder="downloaded_audio")

    # Step 2: Transcribe
    state.transcription = transcribe_audio(audio_path)

    # Step 3: Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(state.transcription)

    # Step 4: Store each chunk as embedding
    for i, chunk in enumerate(chunks):
        key = f"chunk_{i+1}"
        metadata = {"text": chunk, "url": state.youtube_link}
        store.put(namespace, key, metadata, index=["text"])

    return state


def Retrieval_Node(state: RAGState, store: BaseStore, config: RunnableConfig):
    """Retrieves relevant stored text chunks for this video."""
    user_id = config["configurable"]["user_id"]
    namespace = (user_id, state.youtube_link)

    result = store.search(namespace, query=state.messages[-1].content, limit=3)

    # store.search returns a list of results; extract text safely
    retrieved = [r.value["text"] for r in result] if result else []
    state.retrieved_docs = retrieved
    return state


def Chat_Node(state: RAGState, store: BaseStore, config: RunnableConfig):
    """Generates the final answer using the retrieved context."""
    context = "\n".join(state.retrieved_docs) if state.retrieved_docs else "No context found."
    system_message = SystemMessage(content=f"You are a helpful tutor. Use this context:\n{context}")

    state.final_answer = llm_model.invoke({
        "system_message": system_message,
        "messages": [{"role": "user", "content": state.messages[-1].content}]
    }).content

    return state


# ---------------------- GRAPH BUILD ----------------------
graph_builder = StateGraph(RAGState)

graph_builder.add_node("Check_URL_Node", Check_URL_Node)
graph_builder.add_node("Transcribe_Node", Transcribe_Node)
graph_builder.add_node("Retrieval_Node", Retrieval_Node)
graph_builder.add_node("Chat_Node", Chat_Node)

graph_builder.set_entry_point("Check_URL_Node")

graph_builder.add_conditional_edges(
    source="Check_URL_Node",
    path=Check_URL_Node,
    path_map={
        "transcribe": "Transcribe_Node",
        "not_transcribe": "Retrieval_Node"
    }
)

graph_builder.add_edge("Retrieval_Node", "Chat_Node")
graph_builder.set_finish_point("Chat_Node")

# Example placeholders for your actual initialized objects
checkpointer = None  # replace later
store = PostgresStore(conn=Connection)  # placeholder — use your real DB connection

graph = graph_builder.compile(checkpointer=checkpointer, store=store)
