from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes import transcribe, chat

app = FastAPI(
    title="YouTube RAG Chatbot API",
    version="1.0.0",
    description="RAG-based chatbot using LangGraph + Gemini + PostgreSQL"
)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include all route modules
app.include_router(transcribe.router)
app.include_router(chat.router)

@app.get("/")
def root():
    return {"message": "YouTube RAG Chatbot API is running 🚀"}

@app.get("/health")
def health_check():
    return {"status": "ok"}

# You can run directly: uvicorn APIs.main:app --reload
