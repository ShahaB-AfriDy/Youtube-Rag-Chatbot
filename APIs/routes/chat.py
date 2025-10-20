from fastapi import APIRouter
from pydantic import BaseModel
from backend.pipelines.RAG_Bot import app as graph_app, RAGState
from langchain_core.messages import HumanMessage

router = APIRouter(prefix="/chat", tags=["Chat"])

class ChatRequest(BaseModel):
    youtube_url: str
    user_input: str
    user_id: str = "user_001"
    thread_id: str = "1"

@router.post("/")
def chat_with_video(req: ChatRequest):
    """
    Ask a question about a previously transcribed video.
    """
    state = RAGState(
        messages=[HumanMessage(content=req.user_input)],
        youtube_link=req.youtube_url
    )
    config = {"configurable": {"user_id": req.user_id, "thread_id": req.thread_id}}

    result = graph_app.invoke(state, config=config)
    return {
        "status": "success",
        "answer": result.get("final_answer"),
    }
