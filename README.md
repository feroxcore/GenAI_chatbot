


# GenAI_chatbot

AI-powered chatbot that uses Retrieval-Augmented Generation (RAG) for intelligent document Q&A and semantic search.

---

##  Features

- **Multi-format Document Support**: Upload PDFs, Word docs, text files, CSV, Excel, JSON.
- **Smart Text Processing**: Automatically chunks documents and creates semantic embeddings using FAISS.
- **Real-time Chat**: WebSocket-powered chat with typing indicators and immediate responses.
- **Feedback Loop**: Rate AI responses to help improve performance and quality.
- **Robust UI & Logging**: Professional interface, comprehensive error handling, and detailed logs for troubleshooting.

---

##  Quick Start

### Prerequisites

- Python 3.8+
-  API key  
- Basic familiarity with FastAPI and Git

### Setup

```bash
git clone https://github.com/feroxcore/GenAI_chatbot.git
cd GenAI_chatbot
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
Set your API key:

export EURON_API_KEY="your-api-key-here"
Run the server:

uvicorn app:app --host 0.0.0.0 --port 8000 --reload
Open your browser at http://localhost:8000

Usage Overview
1. Upload your documents (PDF, DOCX, TXT, CSV, Excel, JSON).
2. Ask questions in plain English.
3. Get smart answers grounded in your content.
4. Rate the responses to improve the system.

Architecture Snapshot

 User → FastAPI Endpoints → File Upload → Document Chunking → 
 Embeddings (via Euron AI) → FAISS Vector Store → Semantic Search → AI Response
* Backend: FastAPI (Python, async-ready)
* Vector Search: FAISS
* AI Model: text-embedding-3-small  for embedding and gpt-4.1-nano for  answer
* Storage: JSON (for document chunks/embeddings), SQLite (for feedback)

Deployment Options
* Development: uvicorn app:app --reload
* Production: Use Gunicorn or Docker
    * docker-compose.yml included for easy container management
    * Deployable to AWS, Heroku, or other cloud platforms
