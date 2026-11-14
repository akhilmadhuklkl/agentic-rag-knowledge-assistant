# Streamlit backup interface

The Streamlit application is intentionally a thin client. It calls the local FastAPI API and does not implement ingestion, retrieval, orchestration, generation, or verification itself.

## Run locally

Start the API from the repository root:

```powershell
uvicorn agentic_rag.api.app:app --reload
```

Install the optional UI dependency and start Streamlit:

```powershell
python -m pip install -e ".[ui]"
streamlit run streamlit_backup/app.py
```

The API URL can be changed with `ASSISTANT_API_URL`.
