# chat_handler.py
import time
from typing import Callable, Optional

from langchain_core.messages import HumanMessage


def _is_tool_rag_message(msg) -> bool:
    """
    Check if a message is a ToolMessage from the tool_rag tool.

    Based on the actual message structure:
    - type: "tool"
    - name: "tool_rag"
    - tool_call_id: exists (e.g., "call_1")
    - content: contains Document objects with substantial content

    Args:
        msg: Message object to check

    Returns:
        bool: True if this is a tool_rag ToolMessage
    """
    # Check if it's a tool message
    if not (hasattr(msg, "type") and msg.type == "tool"):
        return False

    # Check if it's from tool_rag specifically
    if not (hasattr(msg, "name") and msg.name == "tool_rag"):
        return False

    # Check if it has a tool_call_id (indicates it's a response to a tool call)
    if not (hasattr(msg, "tool_call_id") and msg.tool_call_id):
        return False

    # Check if it has substantial content (tool_rag returns Document objects)
    if not (hasattr(msg, "content") and msg.content and len(msg.content) > 100):
        return False

    # Additional check: tool_rag content typically contains "Document("
    if "Document(" not in msg.content:
        return False

    return True


def process_question_with_agent(
    agent, question, progress_callback: Optional[Callable[[str, int], None]] = None
):
    """
    Process a user question using the RAG agent and return the answer and retrieved documents.

    Args:
        agent: The configured RAG agent
        question (str): The user's question
        progress_callback: Optional callback function for progress updates (status, progress_percent)

    Returns:
        tuple: (answer, was_rag_used, retrieved_docs) - The generated response, whether RAG tool was used, and actual documents used
    """
    # Create the message for the agent
    messages = [HumanMessage(content=question)]

    # Measure execution time (minimal logging for speed)
    start_time = time.time()

    try:
        # Progress callback for starting agent processing
        if progress_callback:
            progress_callback("🚀 Starting agent processing...", 5)

        # Invoke the agent to get the answer (callbacks are handled within agent nodes)
        result = agent.invoke({"messages": messages})

        # Progress callback for extracting results
        if progress_callback:
            progress_callback("📋 Extracting results...", 85)

        answer = (
            result["messages"][-1].content
            if result.get("messages")
            else "No answer generated"
        )

        # Extract the actual documents used by tool_rag
        retrieved_docs = []
        was_rag_used = False

        for msg in result.get("messages", []):
            # Look for ToolMessage from tool_rag using multiple criteria
            if _is_tool_rag_message(msg):
                was_rag_used = True
                # The content of ToolMessage contains the retrieved documents
                # If content is a list of documents, add them individually
                if isinstance(msg.content, list):
                    retrieved_docs.extend(msg.content)
                else:
                    # If content is a string representation of multiple documents, split it
                    if isinstance(msg.content, str) and "Document(" in msg.content:
                        # Split by Document( to get individual documents
                        doc_parts = msg.content.split("Document(")[
                            1:
                        ]  # Skip first empty part
                        for doc_part in doc_parts:
                            # Reconstruct the Document string
                            doc_string = "Document(" + doc_part.rstrip("]").rstrip(",")
                            retrieved_docs.append(doc_string)
                    else:
                        # If content is a single string, add it as is
                        retrieved_docs.append(msg.content)

        # Progress callback for final processing
        if progress_callback:
            if was_rag_used:
                progress_callback("📚 Extracting document sources...", 90)
            else:
                progress_callback("✅ Generating final answer...", 90)

    except Exception as e:
        print(f"[ERROR] Agent processing failed: {e}")
        answer = "Sorry, I cannot answer this question at the moment. Please try again."
        was_rag_used = False
        retrieved_docs = []

    execution_time = time.time() - start_time
    print(f"[PERF] Total time: {execution_time:.2f}s")

    # Final progress callback
    if progress_callback:
        progress_callback("✅ Complete!", 100)

    # Add timing info to the answer
    answer_with_timing = f"{answer}\n\n*Response time: {execution_time:.2f} seconds*"

    return answer_with_timing, was_rag_used, retrieved_docs
