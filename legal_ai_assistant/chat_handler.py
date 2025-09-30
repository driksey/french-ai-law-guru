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

    # Measure execution time with detailed timing
    start_time = time.time()
    print(f"[PERF] Starting agent processing for question: {question[:50]}...")

    try:
        # Invoke the agent to get the answer
        agent_start = time.time()
        result = agent.invoke({"messages": messages})
        agent_time = time.time() - agent_start
        
        answer = result["messages"][-1].content if result.get("messages") else "No answer generated"
        
        print(f"[PERF] Agent processing completed in {agent_time:.2f}s")
        
    except Exception as e:
        print(f"[PERF] Agent processing failed: {e}")
        answer = "Sorry, I cannot answer this question at the moment. Please try again."

    execution_time = time.time() - start_time
    print(f"[PERF] Total execution time: {execution_time:.2f}s")

    # Add timing info to the answer
    answer_with_timing = f"{answer}\n\n*Response time: {execution_time:.2f} seconds*"

    return answer_with_timing
