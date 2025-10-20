Perfect 👍 — here’s your **final, clean, and slightly compact `README.md`** with your LinkedIn link added to the author section:

---

```markdown
# 🎥 YouTube RAG Chatbot using LangGraph

An intelligent chatbot powered by **LangGraph**, **LLMs**, and **PostgreSQL**, designed to extract insights and answer questions from **YouTube videos**.  
Simply paste a video URL and chat with your AI assistant — powered by **retrieval-augmented generation (RAG)**.

---

## 🚀 Features

- 🧠 Modular LangGraph pipeline  
- 🔍 Automatic YouTube transcript retrieval  
- 💾 PostgreSQL-based conversation memory  
- 🗂️ Thread-based chat management  
- 💬 Real-time interactive chat  
- 🎨 Clean and modern UI  

---

## 🧩 Architecture Overview

The chatbot uses a **LangGraph workflow** connecting:
- **Retriever Node** → Fetches YouTube transcripts  
- **Chat Node** → Handles responses & context  
- **Store Node** → Saves chats in PostgreSQL  
- **Frontend + Backend** → Unified interface  

![Graph Flow](Images/Graph-Flow.png)

---

## 🖥️ Application Preview

| Section | Screenshot |
|----------|-------------|
| 🏠 Home | ![Home](Images/Home.png) |
| 🔐 Login | ![Login](Images/Login.png) |
| 🔗 Add URL | ![Add URL](Images/Add-URL.png) |
| 💬 Chat Interface | ![Chat Interface](Images/Chat-Interface.png) |
| 🕓 Chat History | ![Chat History](Images/Chat-History.png) |

---

## 🧱 Folder Structure

```

D:.
├───backend
│   ├───pipelines
│   ├───Testing
│   └───utils
├───frontend
├───Images
└───downloaded_audio

````

---

## ⚙️ Setup Instructions

```bash
# 1. Clone the repo
git clone https://github.com/ShahaB-AfriDy/Youtube-Rag-Chatbot.git
cd Youtube-Rag-Chatbot

# 2. Create and activate virtual environment
python -m venv venv
venv\Scripts\activate  # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup PostgreSQL
CREATE DATABASE youtube_rag;

# 5. Run the app
python main.py
````

---

## 🧠 Powered By

[LangGraph](https://github.com/langchain-ai/langgraph) · [LangChain](https://github.com/langchain-ai/langchain) · [PostgreSQL](https://www.postgresql.org/)
[YouTubeDL](https://github.com/yt-dlp/yt-dlp) · [FastAPI](https://fastapi.tiangolo.com/) · [React](https://react.dev/)

---

## 🌟 Highlights

* LLM integration via graph-based flow
* Persistent multi-session chat memory
* Real-world RAG on YouTube transcripts
* Production-ready modular structure

---

## 👨‍💻 Author

**Muhammad Shahab Afridi**
📦 [GitHub: ShahaB-AfriDy](https://github.com/ShahaB-AfriDy)
🔗 [LinkedIn: Shahab Afridy](https://www.linkedin.com/in/shahab-afridy-9ba965286/)

---

⭐ *If you like this project, please give it a star!*

```

---

Would you like me to add **GitHub badges** (for Python, LangGraph, PostgreSQL, etc.) at the top for a more professional look? It gives it that polished “open-source project” feel.
```
