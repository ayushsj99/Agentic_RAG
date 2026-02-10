from langgraph.graph import MessagesState
from models.gemini_LLM import gemini_model
from models.ollama_LLM import ollama_model
from backend.agent_logger import log



GENERATE_PROMPT = (
    "You are a helpful assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Answer in plain, natural language as if speaking to a person. "
    "Do NOT output JSON, code, or raw data. "
    "Use three sentences maximum and keep the answer concise.\n\n"
    "Question: {question}\n\n"
    "Context:\n{context}\n\n"
    "Answer:"
)

response_model = ollama_model

def generate_answer(state: MessagesState):
    """Generate an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    log("answer_generator", "Generating final answer from retrieved context...")
    log("answer_generator", f"  Question: {question[:150]}")
    log("answer_generator", f"  Context preview: {context[:200]}...")
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = response_model.invoke([{"role": "user", "content": prompt}])
    log("answer_generator", f"  Final answer: {response.content[:300]}")
    return {"messages": [response]}