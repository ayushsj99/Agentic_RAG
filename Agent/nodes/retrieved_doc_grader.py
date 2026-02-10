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
    "You are a helpful document relevance grader.\n\n"
    "User Question: {question}\n\n"
    "Retrieved Context:\n{context}\n\n"
    "TASK: Decide whether the retrieved context is useful for answering the user's question.\n\n"
    "Grade as 'yes' if ANY of the following are true:\n"
    "- The context directly answers the question\n"
    "- The context contains partial information, keywords, names, numbers, or identifiers related to the question\n"
    "- The context is from the same document and could reasonably help infer or locate the answer\n"
    "- The context includes a sentence or phrase that appears relevant, even if the answer is not complete\n\n"
    "Grade as 'no' ONLY if:\n"
    "- The context is completely unrelated to the question\n"
    "- The context discusses an entirely different topic with no overlap in meaning or intent\n\n"
    "Be inclusive rather than strict. If unsure, lean toward 'yes'.\n\n"
    "Binary score: 'yes' or 'no'"
)


class GradeDocuments(BaseModel):
    binary_score: str = Field(
        description="'yes' if any excerpt is relevant, 'no' only if all are unrelated"
    )


grader_model = ollama_model


def grade_documents(state: MessagesState) -> Literal["generate_answer", "rewrite_question", "cannot_answer"]:
    question = _get_last_user_question(state["messages"])
    context = state["messages"][-1].content

    retrieval_count = sum(1 for m in state["messages"] if isinstance(m, ToolMessage))
    log("doc_grader", f"Retrieval loop {retrieval_count}/{MAX_REWRITE_LOOPS}")

    # ALWAYS CHECK LOOP LIMIT FIRST
    if retrieval_count >= MAX_REWRITE_LOOPS:
        log("doc_grader", f"Max loops reached. Generating answer with available context or failing gracefully.")
        
        # Clean the low confidence flag if present
        clean_context = context.replace("[LOW_CONFIDENCE_RETRIEVAL]\n\n", "")
        
        if clean_context and clean_context.strip() and clean_context != "No relevant documents found.":
            log("doc_grader", "Forcing answer generation with best available context")
            return "generate_answer"
        else:
            log("doc_grader", "No usable context found")
            return "cannot_answer"

    # Check low confidence
    if context.startswith("[LOW_CONFIDENCE_RETRIEVAL]"):
        log("doc_grader", "Low confidence -> rewriting question")
        return "rewrite_question"

    # Normal grading
    context_sample = context[:1500]
    log("doc_grader", "Grading retrieved documents for relevance...")
    log("doc_grader", f"  Question: {question[:150]}")
    log("doc_grader", f"  Context sample: {context_sample[:300]}...")

    prompt = GRADE_PROMPT.format(question=question, context=context_sample)
    try:
        response = grader_model.with_structured_output(GradeDocuments).invoke(
            [{"role": "user", "content": prompt}]
        )
        score = response.binary_score.lower().strip()
    except Exception as e:
        log("doc_grader", f"Grading failed ({e}), defaulting to RELEVANT.")
        score = "yes"

    if score == "yes":
        log("doc_grader", "Documents are RELEVANT -> generating answer.")
        return "generate_answer"
    else:
        log("doc_grader", f"Documents NOT relevant -> rewriting (attempt {retrieval_count}/{MAX_REWRITE_LOOPS}).")
        return "rewrite_question"