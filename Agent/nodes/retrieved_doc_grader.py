from pydantic import BaseModel, Field
from typing import Literal
from langchain_core.messages import ToolMessage
from models.gemini_LLM import gemini_model
from models.ollama_LLM import ollama_model
from langgraph.graph import MessagesState
from backend.agent_logger import log

# Maximum number of retrieve -> rewrite loops before forcing an answer
MAX_REWRITE_LOOPS = 5

GRADE_PROMPT = (
    "You are a strict relevance grader for a document retrieval system.\n\n"
    "User question: {question}\n\n"
    "Retrieved document:\n{context}\n\n"
    "Determine if the document contains information that can help answer "
    "the user's question. Consider:\n"
    "- Does it mention the same topic, entities, or concepts?\n"
    "- Does it provide facts, data, or context useful for answering?\n"
    "- Partial relevance still counts as 'yes'.\n\n"
    "Respond with 'yes' if the document is relevant, 'no' if it is completely unrelated."
)


class GradeDocuments(BaseModel):  
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


grader_model = ollama_model


def grade_documents(
    state: MessagesState,
) -> Literal["generate_answer", "rewrite_question", "cannot_answer"]:
    """Determine whether the retrieved documents are relevant to the question."""
    question = state["messages"][0].content
    context = state["messages"][-1].content

    # ── Loop-limit check ────────────────────────────────────────────
    retrieval_count = sum(
        1 for m in state["messages"] if isinstance(m, ToolMessage)
    )
    log("doc_grader", f"Retrieval loop {retrieval_count}/{MAX_REWRITE_LOOPS}")

    if retrieval_count >= MAX_REWRITE_LOOPS:
        log("doc_grader",
            f"Hit max rewrite loops ({MAX_REWRITE_LOOPS}). "
            "Could not find relevant documents -> cannot answer.")
        return "cannot_answer"

    # ── Normal grading ──────────────────────────────────────────────
    log("doc_grader", "Grading retrieved documents for relevance...")
    log("doc_grader", f"  Question: {question[:150]}")
    log("doc_grader", f"  Context preview: {context[:200]}...")
    prompt = GRADE_PROMPT.format(question=question, context=context)
    try:
        response = (
            grader_model
            .with_structured_output(GradeDocuments).invoke(  
                [{"role": "user", "content": prompt}]
            )
        )
        score = response.binary_score
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