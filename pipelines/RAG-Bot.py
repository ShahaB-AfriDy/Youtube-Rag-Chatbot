import os
from dotenv import load_dotenv
from typing import Annotated, Optional, List
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END, add_messages
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from psycopg import Connection


# 🔧 Load environment
load_dotenv()

# --- LLM setup ---
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY")
)
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# --- Database for checkpoints + store ---
user_name = "postgres"
password = "postgres"
database_name = "ragdb"
port = "5432"
DB_URI = f"postgresql://{user_name}:{password}@localhost:{port}/{database_name}?sslmode=disable"

connection_kwargs = {"autocommit": True, "prepare_threshold": 0}
store_conn = Connection.connect(DB_URI, **connection_kwargs)
store = PostgresStore(store_conn)

saver_conn = Connection.connect(DB_URI, **connection_kwargs)
checkpointer = PostgresSaver(saver_conn)

store.setup()
checkpointer.setup()


# --- Shared State Model ---
class RAGState(BaseModel):
    query: Optional[str] = None
    docs: Optional[List[str]] = None
    messages: Annotated[List[AnyMessage], add_messages] = []


# 🧩 Node 1: Load Context / Query
def LoadContextNode(state: RAGState, store: PostgresStore, config: RunnableConfig) -> RAGState:
    print("🟦 LoadContextNode")
    query_msg = state.messages[-1].content
    state.query = query_msg
    return state


# 🧩 Node 2: Retrieve Relevant Documents
def RetrieveDocsNode(state: RAGState, store: PostgresStore, config: RunnableConfig) -> RAGState:
    print("🟩 RetrieveDocsNode")

    # Example: load FAISS index or dynamically build it
    index_path = "vectorstore/faiss_index"
    if not os.path.exists(index_path):
        print("⚠️ Vector store not found! Creating one for demo...")
        sample_docs = [
            Document(page_content="LangGraph is a stateful orchestration library for LLM workflows."),
            Document(page_content="LangChain provides LLM abstractions like Chains, Tools, and Agents."),
        ]
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
        chunks = text_splitter.split_documents(sample_docs)
        vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
        vectorstore.save_local(index_path)

    vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
    retrieved = vectorstore.similarity_search(state.query, k=3)

    state.docs = [doc.page_content for doc in retrieved]
    return state


# 🧩 Node 3: Generate Final Answer
def GenerateAnswerNode(state: RAGState, store: PostgresStore, config: RunnableConfig) -> RAGState:
    print("🟨 GenerateAnswerNode")

    context = "\n\n".join(state.docs or [])
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful AI assistant. Use the provided context to answer accurately."),
        MessagesPlaceholder("messages"),
    ])

    llm_chain = prompt | llm

    system_message = f"Context:\n{context}\n\nAnswer the user query concisely."
    response = llm_chain.invoke({
        "system_message": system_message,
        "messages": [HumanMessage(content=state.query)]
    })

    state.messages.append(AIMessage(content=response.content))
    return state


# --- Build the LangGraph ---
graph_builder = StateGraph(RAGState)
graph_builder.add_node("LoadContextNode", LoadContextNode)
graph_builder.add_node("RetrieveDocsNode", RetrieveDocsNode)
graph_builder.add_node("GenerateAnswerNode", GenerateAnswerNode)

graph_builder.add_edge(START, "LoadContextNode")
graph_builder.add_edge("LoadContextNode", "RetrieveDocsNode")
graph_builder.add_edge("RetrieveDocsNode", "GenerateAnswerNode")
graph_builder.add_edge("GenerateAnswerNode", END)

graph = graph_builder.compile(checkpointer=checkpointer, store=store)


# --- Run interactively ---
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "1", "user_id": "1"}}

    while True:
        user_input = input("Ask a question: ")
        if user_input.lower() in ["exit", "quit", "q"]:
            break

        input_state = RAGState(messages=[HumanMessage(content=user_input)])
        output_state = graph.invoke(input_state, config=config)

        print("🤖 Response:", output_state.messages[-1].content)
