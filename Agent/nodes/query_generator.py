from langgraph.graph import MessagesState
from langchain_core.messages import HumanMessage, AIMessage
from models.gemini_LLM import gemini_model
from models.ollama_LLM import ollama_model
from Agent.nodes.retriever import retriever_tool
from backend.agent_logger import log

response_model = ollama_model

SYSTEM_PROMPT = (
    "You are a helpful assistant with access to a document retrieval tool. "
    "Use the retriever tool when the user asks about specific information that "
    "might be in their uploaded documents. Respond directly for greetings, "
    "general knowledge, or follow-up clarifications. "
    "When you decide to retrieve, formulate a precise, keyword-rich search "
    "query instead of the user's raw conversational question."
)

QUERY_OPT_PROMPT = (
    "Given this user question, generate a concise search query optimized "
    "for semantic similarity search against a document store. "
    "Remove filler words, keep key concepts and entities, expand "
    "abbreviations if any. Output ONLY the optimized search query.\n\n"
    "Question: {question}\n\n"
    "Optimized search query:"
)


def _optimize_search_query(question: str) -> str:
    prompt = QUERY_OPT_PROMPT.format(question=question)
    log("query_generator", "Optimizing search query for retrieval...")
    result = response_model.invoke([{"role": "user", "content": prompt}])
    optimized = result.content.strip()
    if optimized:
        log("query_generator", f"Optimized query: '{optimized[:120]}'")
        return optimized
    return question


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
            original_query = tc["args"].get("query", "")
            if original_query:
                tc["args"]["query"] = _optimize_search_query(original_query)
            log("query_generator", f"  Tool call: {tc['name']}({tc['args']})")
    else:
        log("query_generator", f"LLM decided to RESPOND directly: {response.content[:200]}")

    return {"messages": [response]}