from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.messages import ToolMessage, HumanMessage
from models.ollama_LLM import ollama_model
from langgraph.graph import MessagesState
from backend.agent_logger import log


def _get_last_user_question(messages) -> str:
    """Extract the last user question from messages."""
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content
        if isinstance(msg, dict) and msg.get("role") == "user":
            return msg.get("content", "")
    return messages[0].content if messages else ""

# Maximum number of retrieve -> rewrite loops before forcing an answer
MAX_REWRITE_LOOPS = 5

GRADE_PROMPT = (
    "A user asked: {question}\n\n"
    "Below are excerpts retrieved from their documents:\n{context}\n\n"
    "Do ANY of these excerpts contain information related to the question? "
    "Even a single relevant sentence counts as 'yes'. "
    "Answer 'no' ONLY if every single excerpt is about a completely unrelated topic."
)


class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="'yes' if any excerpt is relevant, 'no' only if all are unrelated"
    )


grader_model = ollama_model


def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question", "cannot_answer"]:
    question = _get_last_user_question(state["messages"])
    context = state["messages"][-1].content

    retrieval_count = sum(
        1 for m in state["messages"] if isinstance(m, ToolMessage)
    )
    log("doc_grader", f"Retrieval loop {retrieval_count}/{MAX_REWRITE_LOOPS}")

    if retrieval_count >= MAX_REWRITE_LOOPS:
        log("doc_grader",
            f"Hit max rewrite loops ({MAX_REWRITE_LOOPS}). "
            "Could not find relevant documents -> cannot answer.")
        return "cannot_answer"


    context_sample = context[:1500]
    log("doc_grader", "Grading retrieved documents for relevance...")
    log("doc_grader", f"  Question: {question[:150]}")
    log("doc_grader", f"  Context sample: {context_sample[:300]}...")

    prompt = GRADE_PROMPT.format(question=question, context=context_sample)
    try:
        response = (
            grader_model
            .with_structured_output(GradeDocuments).invoke(
                [{"role": "user", "content": prompt}]
            )
        )
        score = response.binary_score.lower().strip()
    except Exception as e:
        log("doc_grader", f"Grading failed ({e}), defaulting to RELEVANT.")
        score = "yes"

    if score == "yes":
        log("doc_grader", "Documents are RELEVANT -> generating answer.")
        return "generate_answer"
    else:
        log("doc_grader",
            f"Documents NOT relevant -> rewriting question "
            f"(attempt {retrieval_count}/{MAX_REWRITE_LOOPS}).")
        return "rewrite_question"