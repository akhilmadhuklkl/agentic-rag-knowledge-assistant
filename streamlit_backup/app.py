"""Streamlit backup UI for the Agentic RAG Knowledge Assistant.

Run with:
    streamlit run streamlit_backup/app.py
"""

import os

import streamlit as st

from streamlit_backup.api_client import AssistantApiClient

st.set_page_config(page_title="Agentic RAG Assistant", page_icon="📚")
st.title("📚 Agentic RAG Knowledge Assistant")
st.caption("Backup interface — the API and application services remain the product core.")

client = AssistantApiClient(os.getenv("ASSISTANT_API_URL", "http://localhost:8000"))

with st.sidebar:
    st.header("Knowledge base")
    directory = st.text_input("Local documents directory", value="data/documents")
    if st.button("Index documents"):
        try:
            summary = client.index_directory(directory)
            st.success(
                f"Indexed {summary['source_count']} files and "
                f"{summary['chunk_count']} chunks."
            )
        except Exception as error:  # pragma: no cover - UI error display  # noqa: BLE001
            st.error(f"Indexing failed: {error}")

question = st.text_area("Ask a question", placeholder="What is the refund policy?")
if st.button("Ask assistant", type="primary") and question.strip():
    try:
        result = client.ask(question)
        st.subheader("Answer")
        st.write(result["answer"])
        st.write(f"Verified: {'yes' if result['verified'] else 'no'}")
        st.write(f"Attempts: {result['attempts']} | Duration: {result['duration_ms']} ms")
        if result["citations"]:
            st.subheader("Sources")
            for citation in result["citations"]:
                location = (
                    f"page {citation['page_number']}"
                    if citation["page_number"] is not None
                    else "document"
                )
                st.write(f"[{citation['citation_id']}] {citation['source_name']} ({location})")
    except Exception as error:  # pragma: no cover - UI error display  # noqa: BLE001
        st.error(f"Assistant request failed: {error}")
