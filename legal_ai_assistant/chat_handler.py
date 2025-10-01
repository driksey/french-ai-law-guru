# chat_handler.py
import time

from langchain_core.messages import HumanMessage


def process_question_with_agent(agent, question):
    """
    Process a user question using the RAG agent and return the answer and retrieved documents.

    Args:
        agent: The configured RAG agent
        question (str): The user's question

    Returns:
        tuple: (answer, was_rag_used) - The generated response and whether RAG tool was used
    """
    # Create the message for the agent
    messages = [
        HumanMessage(
            content=question
        )
    ]

    # Measure execution time (minimal logging for speed)
    start_time = time.time()

    try:
        # Invoke the agent to get the answer
        result = agent.invoke({"messages": messages})
        answer = result["messages"][-1].content if result.get("messages") else "No answer generated"

        # Check if tool_rag was used by looking for ToolMessage in the conversation
        was_rag_used = any(
            hasattr(msg, 'tool_call_id') and msg.tool_call_id == "parsed-tool-call"
            for msg in result.get("messages", [])
        )

    except Exception as e:
        print(f"[ERROR] Agent processing failed: {e}")
        answer = "Sorry, I cannot answer this question at the moment. Please try again."
        was_rag_used = False

    execution_time = time.time() - start_time
    print(f"[PERF] Total time: {execution_time:.2f}s")

    # Add timing info to the answer
    answer_with_timing = f"{answer}\n\n*Response time: {execution_time:.2f} seconds*"

    return answer_with_timing, was_rag_used
