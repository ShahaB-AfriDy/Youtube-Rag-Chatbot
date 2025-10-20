Perfect 😎 — here’s your **final polished `README.md`** with stylish **GitHub badges** at the top (Python, LangGraph, PostgreSQL, FastAPI, and React), plus your GitHub and LinkedIn links nicely formatted.

---

```markdown
# 🎥 YouTube RAG Chatbot using LangGraph

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-%F0%9F%A4%96-lightblue.svg)](https://github.com/langchain-ai/langgraph)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-DB-blue.svg?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Backend-success.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-Frontend-61DAFB.svg?logo=react&logoColor=white)](https://react.dev/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

An intelligent chatbot powered by **LangGraph**, **LLMs**, and **PostgreSQL**, designed to extract insights and answer questions from **YouTube videos**.  
Simply paste a video URL and chat with your AI assistant — powered by **retrieval-augmented generation (RAG)**.

---

## 🚀 Features

- 🧠 Modular **LangGraph** pipeline  
- 🔍 Automatic **YouTube transcript retrieval**  
- 💾 Persistent chat memory via **PostgreSQL**  
- 🗂️ Thread-based session management  
- 💬 Real-time, context-aware responses  
- 🎨 Modern **FastAPI + React** interface  

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

* 🧩 LLM integration via graph-based flow
* 💾 Persistent multi-session chat memory
* 🎥 Real-world RAG on YouTube transcripts
* 🧱 Production-ready modular structure

---

## 👨‍💻 Author

**Muhammad Shahab Afridi**
📦 [GitHub: ShahaB-AfriDy](https://github.com/ShahaB-AfriDy)
🔗 [LinkedIn: Shahab Afridy](https://www.linkedin.com/in/shahab-afridy-9ba965286/)

---

⭐ *If you like this project, please give it a star!*

```

---

**What’s new:**
- Added **professional shields.io badges** (Python, LangGraph, PostgreSQL, FastAPI, React, License)  
- Kept your text clean, clear, and concise  
- Maintained a professional GitHub aesthetic  

Would you like me to also generate a **project banner image (hero header)** that you can place at the top of the README (like `Images/Banner.png`)? It would make your project visually stand out on GitHub.
```
