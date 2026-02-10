from langchain.messages import HumanMessage
from models.gemini_LLM import gemini_model
from models.ollama_LLM import ollama_model
from langgraph.graph import MessagesState
from backend.agent_logger import log

REWRITE_PROMPT = (
    "You are a query rewriter for a document retrieval system. "
    "The previous search did not return relevant results.\n\n"
    "Original question:\n{question}\n\n"
    "Rewrite this into a better search query by:\n"
    "- Using different keywords or synonyms\n"
    "- Being more specific about the core topic\n"
    "- Removing vague or conversational language\n"
    "- Keeping it concise (under 20 words)\n\n"
    "Output ONLY the rewritten query, nothing else."
)

response_model = ollama_model

def rewrite_question(state: MessagesState):
    """Rewrite the original user question."""
    messages = state["messages"]
    question = messages[0].content
    log("rewriter", f"Rewriting question: {question[:120]}")
    prompt = REWRITE_PROMPT.format(question=question)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    log("rewriter", f"Rewritten to: {response.content[:120]}")
    return {"messages": [HumanMessage(content=response.content)]}