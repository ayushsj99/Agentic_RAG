from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import AIMessage

from Agent.nodes.retrieved_doc_grader import grade_documents
from Agent.nodes.question_rewriter import rewrite_question
from Agent.nodes.final_ans_generator import generate_answer
from Agent.nodes.query_generator import generate_query_or_respond
from Agent.nodes.retriever import retriever_tool
from langgraph.graph import MessagesState
from backend.agent_logger import log as agent_log


def cannot_answer(state: MessagesState):
    """Return a polite 'could not answer' message when the grader
    cannot find relevant documents after all rewrite attempts."""
    agent_log("cannot_answer",
              "No relevant documents found after max retries. "
              "Returning 'unable to answer' to user.")
    return {
        "messages": [
            AIMessage(
                content=(
                    "I'm sorry, I could not find relevant information "
                    "in the available documents to answer your question. "
                    "Please try rephrasing your question or upload "
                    "additional documents that may contain the answer."
                )
            )
        ]
    }


workflow = StateGraph(MessagesState)

# Define the nodes we will cycle between
workflow.add_node(generate_query_or_respond)
workflow.add_node("retrieve", ToolNode([retriever_tool]))
workflow.add_node(rewrite_question)
workflow.add_node(generate_answer)
workflow.add_node(cannot_answer)

workflow.add_edge(START, "generate_query_or_respond")

# Decide whether to retrieve
workflow.add_conditional_edges(
    "generate_query_or_respond",
    # Assess LLM decision (call `retriever_tool` tool or respond to the user)
    tools_condition,
    {
        # Translate the condition outputs to nodes in our graph
        "tools": "retrieve",
        END: END,
    },
)

# Edges taken after the `action` node is called.
workflow.add_conditional_edges(
    "retrieve",
    # Assess agent decision
    grade_documents,
)
workflow.add_edge("generate_answer", END)
workflow.add_edge("cannot_answer", END)
workflow.add_edge("rewrite_question", "generate_query_or_respond")

GRAPH_RECURSION_LIMIT = 30

# Compile
graph = workflow.compile()


def run_conversation():
    """Run an interactive conversation with the RAG agent."""
    print("=" * 70)
    print("Welcome to the Agentic RAG Chatbot!")
    print("=" * 70)
    print("Ask questions about your documents. Type 'quit' or 'exit' to end.\n")
    
    # Initialize conversation history
    conversation_history = []
    
    while True:
        # Get user input
        user_input = input("You: ").strip()
        
        # Check for exit commands
        if user_input.lower() in ['quit', 'exit', 'q']:
            print("\nGoodbye! Thanks for chatting.")
            break
        
        # Skip empty inputs
        if not user_input:
            continue
        
        # Add user message to history
        conversation_history.append({
            "role": "user",
            "content": user_input
        })
        
        print("\nAssistant: ", end="", flush=True)
        
        try:
            # Stream the response and collect final state
            final_state = None
            for chunk in graph.stream(
                {"messages": conversation_history},
                stream_mode="values"
            ):
                final_state = chunk
            
            # Extract the final assistant message
            if final_state and final_state.get("messages"):
                # Get the last message which should be the final answer
                last_message = final_state["messages"][-1]
                if hasattr(last_message, 'content'):
                    print(last_message.content)
                    # Update conversation history
                    conversation_history = final_state["messages"]
            
            print()
            
            # Update conversation history with the full response
            # The graph should have updated the messages list
            
        except KeyboardInterrupt:
            print("\n\nConversation interrupted.")
            break
        except Exception as e:
            print(f"\n\nError: {e}")
            print("Please try again.\n")


def run_single_query(query: str):
    """Run a single query through the RAG agent."""
    print("=" * 70)
    print(f"Query: {query}")
    print("=" * 70)
    
    for chunk in graph.stream(
        {
            "messages": [
                {
                    "role": "user",
                    "content": query,
                }
            ]
        }
    ):
        for node, update in chunk.items():
            print(f"\n--- Update from node: {node} ---")
            if update.get("messages"):
                update["messages"][-1].pretty_print()
            print()


if __name__ == "__main__":
    # You can choose to run either:
    # 1. Interactive conversation mode
    run_conversation()
    
    # 2. Or single query mode (comment out run_conversation() and uncomment below)
    # run_single_query("What does the document say about machine learning?")