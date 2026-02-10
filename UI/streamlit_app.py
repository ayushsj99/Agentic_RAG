import os
import shutil
import streamlit as st

from Agent.agent import graph, GRAPH_RECURSION_LIMIT
from langgraph.errors import GraphRecursionError
from backend.ingestion_pipeline import ingest_document
from backend.agent_logger import log, get_logs, clear as clear_logs
from database.reset_chroma import reset_chroma_db

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "backend", "data")
DATA_DIR = os.path.abspath(DATA_DIR)
os.makedirs(DATA_DIR, exist_ok=True)

# ── Session-state defaults ──────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "stop_agent" not in st.session_state:
    st.session_state.stop_agent = False
if "agent_running" not in st.session_state:
    st.session_state.agent_running = False

st.set_page_config(page_title="Agentic RAG", page_icon="🤖", layout="centered")

# ── Sidebar: Document Upload ────────────────────────────────────────
with st.sidebar:
    st.header("📄 Documents")

    uploaded_file = st.file_uploader(
        "Upload a document",
        type=["pdf", "txt", "docx", "pptx", "xlsx"],
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

    # Reset button
    if st.button("🗑️ Reset Chat & Database", use_container_width=True):
        try:
            removed = reset_chroma_db()
            # Delete all files in the data folder
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


# ── Main Area: Chat ─────────────────────────────────────────────────
st.title("🤖 Agentic RAG Chat")

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Stop button (visible only while agent is running) ───────────────
def _request_stop():
    st.session_state.stop_agent = True

if st.session_state.agent_running:
    st.button("🛑 Stop Agent", on_click=_request_stop, type="primary",
              use_container_width=True, key="stop_btn")

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Reset stop flag for new query
    st.session_state.stop_agent = False
    st.session_state.agent_running = True

    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant"):
        log_container = st.status("Agent is thinking...", expanded=True)
        stopped_early = False
        try:
            clear_logs()
            log("agent", f"Received query: {prompt[:120]}")

            messages = [
                {"role": m["role"], "content": m["content"]}
                for m in st.session_state.messages
                if m["role"] in ("user", "assistant")
            ]
            # Keep only the last 4 messages (2 turns) as context for the agent
            # Full history is still shown in the UI
            MAX_CONTEXT_MESSAGES = 4
            messages = messages[-MAX_CONTEXT_MESSAGES:]

            # Run the agent graph with recursion limit
            config = {"recursion_limit": GRAPH_RECURSION_LIMIT}
            final_state = None
            for chunk in graph.stream(
                {"messages": messages},
                config=config,
                stream_mode="values",
            ):
                final_state = chunk

                # Check if user requested stop
                if st.session_state.stop_agent:
                    log("agent", "User requested STOP. Halting agent.")
                    stopped_early = True
                    break

            reply = ""
            if final_state and final_state.get("messages"):
                last = final_state["messages"][-1]
                if hasattr(last, "content"):
                    reply = last.content

            if stopped_early:
                log("agent", "Agent stopped by user.")
                reply = reply or "_Agent was stopped before generating a complete answer._"
            else:
                log("agent", "Done.")

            logs = get_logs()

            # Show agent steps inside the status widget
            for entry in logs:
                log_container.write(
                    f"**`{entry['time']}`  [{entry['step']}]**  {entry['message']}"
                )

            if stopped_early:
                log_container.update(label="Agent stopped by user",
                                     state="error", expanded=False)
            else:
                log_container.update(label="Agent finished",
                                     state="complete", expanded=False)
        except GraphRecursionError:
            log("agent", "Hit hard recursion limit. Returning 'unable to answer'.")
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
        finally:
            st.session_state.agent_running = False
            st.session_state.stop_agent = False

        st.markdown(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
