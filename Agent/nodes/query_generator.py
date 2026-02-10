import re
from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, AIMessage
from models.ollama_LLM import ollama_model
from Agent.nodes.retriever import retriever_tool
from backend.agent_logger import log

response_model = ollama_model

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a document retrieval tool. "
    "Use the retriever tool when the user asks about specific information that "
    "might be in their uploaded documents. Respond directly for greetings, "
    "general knowledge, or follow-up clarifications. "
    "When you decide to retrieve, pass the user's question directly as the "
    "search query without modifying it."
)


def _sanitize_query(text: str) -> str:
    """Strip any JSON artifacts the LLM might inject into tool-call args."""
    text = re.sub(r'[{}\[\]"\':]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def generate_query_or_respond(state: MessagesState):
    question = state["messages"][-1].content if hasattr(state["messages"][-1], "content") else str(state["messages"][-1])
    log("query_generator", f"Deciding whether to retrieve or respond for: {question[:120]}")

    enriched_messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in state["messages"][:-1]:
        if isinstance(m, HumanMessage):
            enriched_messages.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            enriched_messages.append({"role": "assistant", "content": m.content})
    enriched_messages.append({"role": "user", "content": question})

    response = response_model.bind_tools([retriever_tool]).invoke(enriched_messages)

    has_tool_calls = bool(getattr(response, "tool_calls", None))
    if has_tool_calls:
        log("query_generator", "LLM decided to RETRIEVE documents.")
        for tc in response.tool_calls:
            raw_query = tc["args"].get("query", "")
            clean_query = _sanitize_query(raw_query)
            if clean_query:
                tc["args"]["query"] = clean_query
            log("query_generator", f"  Tool call: {tc['name']}({tc['args']})")
    else:
        log("query_generator", f"LLM decided to RESPOND directly: {response.content[:200]}")

    return {"messages": [response]}