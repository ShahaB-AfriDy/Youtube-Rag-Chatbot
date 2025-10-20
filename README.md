Got it 👍 — you want to **keep your original README layout** (with individual image previews for each section, not in a table),
but update it slightly to mention that your **API is built with FastAPI** and your **frontend uses Next.js**, plus keep it clean and professional.

Here’s your **final refined version** (slightly trimmed, badges kept minimal, individual previews intact, and accurate tech stack):

---

```markdown
# 🎥 YouTube RAG Chatbot using LangGraph

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-%F0%9F%A4%96-lightblue.svg)](https://github.com/langchain-ai/langgraph)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-DB-blue.svg?logo=postgresql&logoColor=white)](https://www.postgresql.org/)
[![FastAPI](https://img.shields.io/badge/API-FastAPI-success.svg?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Next.js](https://img.shields.io/badge/Frontend-Next.js-black.svg?logo=nextdotjs&logoColor=white)](https://nextjs.org/)

---

An intelligent chatbot powered by **LangGraph**, **LLMs**, and **PostgreSQL**, designed to extract insights and answer questions from **YouTube videos**.  
Simply paste a video URL and chat with your AI assistant — powered by **retrieval-augmented generation (RAG)**.

---

## 🚀 Features

- 🧠 **LangGraph-powered pipeline** for modular workflows  
- 🔍 **Automatic YouTube transcript retrieval**  
- 💾 **Persistent conversation memory** via PostgreSQL  
- 🗂️ **Thread-based chat management**  
- 💬 **Real-time context-aware chat**  
- ⚙️ **Backend built with FastAPI**  
- 🌐 **Frontend built with Next.js (React)**  

---

## 🧩 Architecture Overview

The chatbot uses a **LangGraph workflow** connecting:
- A **Retriever Node** → Fetches transcripts from YouTube  
- A **Chat Node** → Handles conversation & context  
- A **Store Node** → Saves threads and responses to PostgreSQL  
- **Frontend (Next.js)** and **Backend (FastAPI)** for smooth interaction  

![Graph Flow](Images/Graph-Flow.png)

---

## 🖥️ Application Preview

### 🏠 Home Page
A simple and elegant landing page to begin chatting.

![Home Page](Images/Home.png)

---

### 🔐 Login Page
Secure access and user management.

![Login](Images/Login.png)

---

### 🔗 Add YouTube URL
Paste any YouTube video link to load and process its transcript.

![Add URL](Images/Add-URL.png)

---

### 💬 Chat Interface
Interactive chatbot with message streaming and context awareness.

![Chat Interface](Images/Chat-Interface.png)

---

### 🕓 Chat History
View previous chat sessions and revisit past insights.

![Chat History](Images/Chat-History.png)

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

### 1. Clone the Repository
```bash
git clone https://github.com/ShahaB-AfriDy/Youtube-Rag-Chatbot.git
cd Youtube-Rag-Chatbot
````

### 2. Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate  # for Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Setup PostgreSQL

```sql
CREATE DATABASE youtube_rag;
```

### 5. Run the App

```bash
python main.py
```

---

## 🧠 Powered By

* [LangGraph](https://github.com/langchain-ai/langgraph)
* [LangChain](https://github.com/langchain-ai/langchain)
* [PostgreSQL](https://www.postgresql.org/)
* [YouTubeDL](https://github.com/yt-dlp/yt-dlp)
* [FastAPI](https://fastapi.tiangolo.com/)
* [Next.js (React)](https://nextjs.org/)

---

## 🌟 Highlights

* **LLM integration** via structured LangGraph flow
* **Persistent memory** for multi-session chats
* **RAG applied to YouTube transcripts**
* **Production-ready architecture** using FastAPI + Next.js

---

## 👨‍💻 Author

**Muhammad Shahab Afridi**
📦 [GitHub: ShahaB-AfriDy](https://github.com/ShahaB-AfriDy)
🔗 [LinkedIn: Shahab Afridy](https://www.linkedin.com/in/shahab-afridy-9ba965286/)

---

⭐ *If you like this project, please give it a star!*

```

---

✅ **Changes made for accuracy & clarity:**
- Kept your **original image-based layout** (for each page preview).  
- Added proper credit: **Backend → FastAPI**, **Frontend → Next.js (React)**.  
- Maintained all structure, just simplified text a bit.  
- Added clean badges at the top (lightweight, professional).  

Would you like me to also create a **short GitHub description and tags** (the one-line text that appears under your repo name on GitHub)?
```
