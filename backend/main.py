from langchain_core.messages import HumanMessage
from pipelines.RAG_Bot import RAGState,app
if __name__ == "__main__":
    # URL = "https://www.youtube.com/shorts/6wHscF7GE6A"
    # URL = "https://www.youtube.com/shorts/x2VefKXyLko"
    # URL = "https://www.youtube.com/shorts/x2VefKXyLko"
    # URL = "https://www.youtube.com/shorts/FH1AMAKgdn4"
    # URL = "https://www.youtube.com/watch?v=wjZofJX0v4M"
    # URL = "https://www.youtube.com/shorts/je4Q1vBCpok"
    # URL = "https://www.youtube.com/shorts/XJ1yWRwZ6JQ"
    # URL = "https://www.youtube.com/shorts/RECOMngbA6Y"
    URL = "https://www.youtube.com/shorts/fiPPR5ZzUO4"
    

    state = RAGState(
        messages=[HumanMessage(content="which is talking in the video?")],
        youtube_link=URL,
        retrieved_docs=[] 
    )

    config = {"configurable": {
            "user_id": "user_001",
            "thread_id": "1",
    }}

    result = app.invoke(state, config=config)
    print(result["final_answer"])


    # close_connections()