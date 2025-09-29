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
        str: The generated response with timing information
    """
    # Create the message for the agent
    messages = [
        HumanMessage(
            content=question
        )
    ]

    # Measure execution time
    start_time = time.time()

    try:
        # Invoke the agent to get the answer
        result = agent.invoke({"messages": messages})
        answer = result["messages"][-1].content if result.get("messages") else "No answer generated"
    except Exception:
        answer = "Sorry, I cannot answer this question at the moment. Please try again."

    execution_time = time.time() - start_time

    # Add timing info to the answer
    answer_with_timing = f"{answer}\n\n*Response time: {execution_time:.2f} seconds*"

    return answer_with_timing
