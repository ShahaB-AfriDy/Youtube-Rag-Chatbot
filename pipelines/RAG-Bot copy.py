from typing import List, Optional, Annotated
from pydantic import BaseModel, Field
from psycopg import Connection

from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.vectorstores.pgvector import PGVector
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_core.messages import AnyMessage, HumanMessage,SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain.schema.runnable import RunnableLambda

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import add_messages
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.store.base import BaseStore

#########################################################
from utils.audio import extract_audio
from pipelines.transcription import transcribe_audio

#########################################################


from dotenv import load_dotenv
import os
load_dotenv()


model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"))

prompt_template = ChatPromptTemplate.from_messages([
    ("system", "{system_message}"),
    MessagesPlaceholder("messages")])


llm_model = prompt_template | model

class RAGState(BaseModel):
    messages: Annotated[List[AnyMessage], add_messages]
    youtube_link = str
    transcription : str
    retrieved_docs: List[Document]
    summarized_context: str
    final_answer: str

#### conditional Node
def Check_URL_Node(state: RAGState, store: BaseStore, config: RunnableConfig) -> RAGState:
    user_id = config["configurable"]["user_id"]
    namespace = (user_id, "memories")
    url_exist = store.search(namespace,filter={"url":state.youtube_link})
    return "transcribe" if url_exist else "not_transcribe"


def Transcribe_Node(state: RAGState, store: BaseStore, config: RunnableConfig) -> RAGState:
    ### extract Audio (download)
    audio_path = extract_audio(state.youtube_link, output_folder="downloaded_audio")
    ### transcription 
    state.transcription = transcribe_audio(audio_path)
    ### store.put
    # ---- SPLIT LONG TEXT INTO CHUNKS ----

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_text(state.transcription)

    # ---- STORE EACH CHUNK AS EMBEDDING ----
    for i, chunk in enumerate(chunks):
        key = f"chunk_{i+1}"
        metadata = {"text": chunk,"url":state.youtube_link}
        metadata = {"text": chunk}
        ## or add logic url should be in the namespace 
        store.put(namespace, key, metadata, index=["text"])

    return state


def Retrieval_Node(state: RAGState, store: BaseStore, config: RunnableConfig) -> RAGState:

    user_id = config["configurable"]["user_id"]
    namespace = (user_id, "memories")
    result = store.search(namespace,query=state.messages[-1].content,limit=3)
    state.retrieved_docs = result.value['text']
    return state

def Chat_Node(state: RAGState, store: BaseStore, config: RunnableConfig) -> RAGState:

    system_message = SystemMessage(content=f"you are the retrieval system and this {state.retrieved_docs}")
    state.final_answer = llm_model.invoke({
        "system_message": system_message,
        "messages": [{"role": "user", "content": state.messages[-1].content}]
    })
    return state


graph_builder = StateGraph(RAGState)

graph_builder.add_node(node="Check_URL_Node", action=lambda state:state)
graph_builder.add_node(node="Transcribe_Node", action=Transcribe_Node)
graph_builder.add_node(node="Retrieval_Node", action=Retrieval_Node)
graph_builder.add_node(node="Chat_Node", action=Chat_Node)

graph_builder.set_entry_point(key="Check_URL_Node")

graph_builder.add_conditional_edges(
    source="Check_URL_Node",
    path=Check_URL_Node,
    path_map={
        "transcribe": "Transcribe_Node",
        "not_transcribe": "Retrieval_Node"
    }
)

graph_builder.add_edge(start_key="Retrieval_Node",end_key="Chat_Node")

graph_builder.set_finish_point(key="Chat_Node")

graph = graph_builder.compile(checkpointer=checkpointer, store=store)