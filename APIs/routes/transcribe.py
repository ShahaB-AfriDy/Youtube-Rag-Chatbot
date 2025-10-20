from fastapi import APIRouter
from pydantic import BaseModel
from backend.pipelines.RAG_Bot import app as graph_app, RAGState

router = APIRouter(prefix="/transcribe", tags=["Transcription"])

class TranscribeRequest(BaseModel):
    youtube_url: str
    user_id: str = "user_001"
    thread_id: str = "1"

@router.post("/")
def transcribe_video(req: TranscribeRequest):
    """
    Run the pipeline to download, transcribe, and store a YouTube video.
    """
    state = RAGState(messages=[], youtube_link=req.youtube_url)
    config = {"configurable": {"user_id": req.user_id, "thread_id": req.thread_id}}

    result = graph_app.invoke(state, config=config)
    return {
        "status": "success",
        "message": "Video processed successfully.",
        "final_answer": result.get("final_answer")
    }
