import re
from langchain.messages import HumanMessage
from models.ollama_LLM import ollama_model
from langgraph.graph import MessagesState
from backend.agent_logger import log

REWRITE_PROMPT = (
    "The previous document search for the question below did not return "
    "good results. Rewrite it as a better search query using different "
    "words or synonyms. Keep the same meaning and topic.\n\n"
    "Original: {question}\n\n"
    "Write ONLY the rewritten query as plain English words. "
)

response_model = ollama_model


def _sanitize(text: str) -> str:
    """Strip JSON-like artifacts that llama3.1 tends to inject."""
    text = re.sub(r'[{}\[\]"\':]+', ' ', text)
    text = re.sub(r'->.*', '', text)           # remove "-> ..." echo
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def rewrite_question(state: MessagesState):
    messages = state["messages"]
    question = messages[0].content
    log("rewriter", f"Rewriting question: {question[:120]}")
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    rewritten = _sanitize(response.content)
    if not rewritten or len(rewritten) < 5:
        log("rewriter", "Rewrite returned bad output, keeping original question.")
        return {"messages": [HumanMessage(content=question)]}
    log("rewriter", f"Rewritten to: {rewritten[:120]}")
    return {"messages": [HumanMessage(content=rewritten)]}