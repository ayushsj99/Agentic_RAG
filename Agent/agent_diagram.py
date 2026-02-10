from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AIMessage
from langgraph.graph import MessagesState

# ---- Import nodes ----
from Agent.nodes.retrieved_doc_grader import grade_documents
from Agent.nodes.question_rewriter import rewrite_question
from Agent.nodes.final_ans_generator import generate_answer
from Agent.nodes.query_generator import generate_query_or_respond
from Agent.nodes.retriever import retriever_tool


def cannot_answer(state: MessagesState):
    """Terminal fallback node"""
    return {
        "messages": [
            AIMessage(
                content=(
                    "I'm sorry, I could not find relevant information "
                    "in the available documents to answer your question."
                )
            )
        ]
    }


def build_graph():
    workflow = StateGraph(MessagesState)

    # ---- Nodes ----
    workflow.add_node("generate_query_or_respond", generate_query_or_respond)
    workflow.add_node("retrieve", ToolNode([retriever_tool]))
    workflow.add_node("grade_documents", grade_documents)
    workflow.add_node("rewrite_question", rewrite_question)
    workflow.add_node("generate_answer", generate_answer)
    workflow.add_node("cannot_answer", cannot_answer)

    # ---- Edges ----
    workflow.add_edge(START, "generate_query_or_respond")

    # Decide whether retrieval is needed
    workflow.add_conditional_edges(
        "generate_query_or_respond",
        tools_condition,
        {
            "tools": "retrieve",
            END: "generate_answer",
        },
    )

    # Retrieval → grading
    workflow.add_edge("retrieve", "grade_documents")

    # Grading decisions
    workflow.add_conditional_edges(
        "grade_documents",
        grade_documents,
        {
            "generate_answer": "generate_answer",
            "rewrite_question": "rewrite_question",
            "cannot_answer": "cannot_answer",
        },
    )

    # Rewrite → retry retrieval (🔁)
    workflow.add_edge("rewrite_question", "retrieve")

    # Terminals
    workflow.add_edge("generate_answer", END)
    workflow.add_edge("cannot_answer", END)

    return workflow.compile()


if __name__ == "__main__":
    app = build_graph()

    png_bytes = app.get_graph().draw_mermaid_png()
    import os
    images_dir = os.path.join(os.path.dirname(__file__), '..', 'images')
    os.makedirs(images_dir, exist_ok=True)
    image_path = os.path.join(images_dir, "agentic_rag_graph.png")
    with open(image_path, "wb") as f:
        f.write(png_bytes)

    print(f"✅ Agentic RAG graph exported as {image_path}")
