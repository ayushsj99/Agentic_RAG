import os
import shutil
import streamlit as st

from Agent.agent import graph, GRAPH_RECURSION_LIMIT
from langgraph.errors import GraphRecursionError
from backend.ingestion_pipeline import ingest_document
from backend.agent_logger import log, get_logs, clear as clear_logs
from database.reset_chroma import reset_chroma_db
from backend.exceptions import check_ollama_health, check_chroma_health

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "backend", "data")
DATA_DIR = os.path.abspath(DATA_DIR)
os.makedirs(DATA_DIR, exist_ok=True)

if "messages" not in st.session_state:
    st.session_state.messages = []
if "services_checked" not in st.session_state:
    st.session_state.services_checked = False

st.set_page_config(page_title="Agentic RAG", page_icon="🤖", layout="centered")


with st.sidebar:
    st.header("📄 Documents")
    
    with st.expander("🔧 Service Status", expanded=not st.session_state.services_checked):
        ollama_ok = check_ollama_health()
        chroma_ok = check_chroma_health()
        st.session_state.services_checked = True
        
        if ollama_ok:
            st.success("Ollama: Connected")
        else:
            st.error("Ollama: Not available")
        
        if chroma_ok:
            st.success("ChromaDB: Connected")
        else:
            st.warning("ChromaDB: Not initialized")

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "txt", "docx", "pptx", "xlsx", "csv", "md"],
    )
    if uploaded_file and st.button("Upload & Ingest"):
        existing_files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
        if existing_files:
            st.warning("A document is already loaded. Please reset the database first before uploading a new one.")
        else:
            file_path = os.path.join(DATA_DIR, uploaded_file.name)
            with st.spinner("Uploading and ingesting..."):
                try:
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    ingest_document(file_path)
                    st.success(f"'{uploaded_file.name}' uploaded and ingested successfully.")
                except Exception as e:
                    st.error(f"Ingestion failed: {e}")

    st.divider()

    if st.button("🗑️ Reset Chat & Database", use_container_width=True):
        try:
            removed = reset_chroma_db()
            for f in os.listdir(DATA_DIR):
                path = os.path.join(DATA_DIR, f)
                if os.path.isfile(path):
                    os.remove(path)
            st.session_state.messages = []
            st.success(f"Reset complete. Removed {removed} vectors and all uploaded files.")
        except Exception as e:
            st.error(f"Reset failed: {e}")

    st.divider()
    st.subheader("Uploaded Files")
    files = [f for f in os.listdir(DATA_DIR) if os.path.isfile(os.path.join(DATA_DIR, f))]
    if files:
        for d in files:
            st.write(f"• {d}")
    else:
        st.caption("No documents yet.")



st.title("🤖 Agentic RAG Chat")

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Ask a question about your documents..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        log_container = st.status("Agent is thinking...", expanded=True)
        try:
            clear_logs()
            log("agent", f"Received query: {prompt[:120]}")

            messages = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
                if m["role"] in ("user", "assistant")
            ]
            MAX_CONTEXT_MESSAGES = 4
            messages = messages[-MAX_CONTEXT_MESSAGES:]

            config = {"recursion_limit": GRAPH_RECURSION_LIMIT}
            final_state = None
            for chunk in graph.stream(
                {"messages": messages},
                config=config,
                stream_mode="values",
            ):
                final_state = chunk

            reply = ""
            if final_state and final_state.get("messages"):
                last = final_state["messages"][-1]
                if hasattr(last, "content"):
                    reply = last.content

            log("agent", "Done.")
            logs = get_logs()
            for entry in logs:
                log_container.write(
                    f"**`{entry['time']}`  [{entry['step']}]**  {entry['message']}"
                )
            log_container.update(label="Agent finished",
                                 state="complete", expanded=False)

        except GraphRecursionError:
            log("agent", "Hit hard recursion limit.")
            reply = (
                "I'm sorry, I could not find relevant information "
                "in the available documents to answer your question. "
                "Please try rephrasing your question or upload "
                "additional documents that may contain the answer."
            )
            logs = get_logs()
            for entry in logs:
                log_container.write(
                    f"**`{entry['time']}`  [{entry['step']}]**  {entry['message']}"
                )
            log_container.update(label="Agent could not answer",
                                 state="error", expanded=False)
        except Exception as e:
            reply = f"Error: {e}"
            log_container.update(label="Error", state="error", expanded=False)

        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
