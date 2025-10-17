import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableConfig
from langchain_core.messages import AnyMessage
from langgraph.graph import StateGraph, START,add_messages,END
from langgraph.checkpoint.postgres import PostgresSaver
from langgraph.store.postgres import PostgresStore
from langgraph.store.base import BaseStore
from trustcall import create_extractor
from langchain_core.messages import HumanMessage,SystemMessage
from pydantic import BaseModel,Field
from typing import Annotated,Optional, List
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from psycopg import Connection

import os
from dotenv import load_dotenv
load_dotenv()

model = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    api_key=os.getenv("GOOGLE_API_KEY"))

user_id = "1"
namespace_for_memory = (user_id, "memories")

user_name = "postgres"
password = "postgres"
database_name = "rag"
port = "5432"

DB_URI = f"postgresql://{user_name}:{password}@localhost:{port}/{database_name}?sslmode=disable"

connection_kwargs = {
    "autocommit": True, ## each change saves immediately
    "prepare_threshold": 0, ## don’t use prepared statements
}
store_conn = Connection.connect(DB_URI, **connection_kwargs)
store = PostgresStore(store_conn)

saver_conn = Connection.connect(DB_URI, **connection_kwargs)
checkpointer = PostgresSaver(saver_conn)

store.setup()
checkpointer.setup()


prompt_template = ChatPromptTemplate.from_messages([
    ("system", "{system_message}"),
    MessagesPlaceholder("messages")])

llm_model = prompt_template | model 


class RAGState(BaseModel):
    messages: Annotated[List[AnyMessage], add_messages]
    retrieved_docs: Optional[List[Document]] = None
    summarized_context: Optional[str] = None
    final_answer: Optional[str] = None


def RetrieveChunks(state: RAGState) -> RAGState:
    print("🧠 [RetrieveChunks] Running...")
    user_question = state.messages[-1].content  # last user message
    vectorstore = get_pgvector_vectorstore()
    results = vectorstore.similarity_search(user_question, k=3)
    state.retrieved_docs = results
    return state



def ChatNode(state: RAGState, store: BaseStore, config: RunnableConfig) -> RAGState:

    user_id = config["configurable"]["user_id"]
    namespace = (user_id, "memories")
    profile = store.get(namespace, key="profile")
    context = str(profile.value) if profile else ""

    system_message = f"You are an assistant. Use context from previous communication: {context}"

    state.messages = llm_model.invoke({
        "system_message": system_message,
        "messages": state.messages[-1:]
    })
    return state




graph_builder = StateGraph(State)
graph_builder.add_node("ChatNode", ChatNode)
graph_builder.add_node("GetProfileNode", GetProfileNode)
graph_builder.add_edge(START, "ChatNode")
graph_builder.add_edge("ChatNode", "GetProfileNode")
graph_builder.add_edge("GetProfileNode", END)
graph = graph_builder.compile(checkpointer=checkpointer, store=store)


if __name__ =="__main__":
    config = {"configurable": {"thread_id":"1","user_id":"1"}}
    while True:
        user_input = input("Query: ")
        if user_input in ["q","quit","exit","break"]:
            print("Bot is terminated!".center(100,"*"))
            break
        input_state = State(messages=[HumanMessage(
        content=user_input)])

        response_state = graph.invoke(input_state, config=config)
        print("printing")
        for message in response_state["messages"][-2:]:
            message.pretty_print()


