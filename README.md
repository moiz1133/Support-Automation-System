# Support Automation System

A scaffold for an AI-powered customer support automation service. It exposes a FastAPI backend that will use OpenAI models and a ChromaDB vector store to answer customer questions by retrieving relevant articles from a local knowledge base (`data/docs/`), with structured JSON logging for observability.

## Running locally

1. Create and activate a virtual environment:
   ```
   python -m venv venv
   venv\Scripts\activate
   ```
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Copy `.env.example` to `.env` and fill in your values:
   ```
   copy .env.example .env
   ```
4. Run the development server:
   ```
   uvicorn app.main:app --reload
   ```
5. Visit `http://127.0.0.1:8000/health` to confirm the app is running.
