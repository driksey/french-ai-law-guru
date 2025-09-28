# chat_handler.py
import time
import re
from langchain_core.messages import HumanMessage
from faq_chatbot.utils import retrieve_documents


def clean_answer(answer):
    """
    Clean the answer from unwanted formatting tags and artifacts.
    
    Args:
        answer (str): Raw answer from the agent
        
    Returns:
        str: Cleaned answer
    """
    if not answer:
        return answer
    
    # Remove common formatting tags
    patterns_to_remove = [
        r'\[/USER\]',           # Remove [/USER] tags
        r'\[/ASSISTANT\]',      # Remove [/ASSISTANT] tags
        r'\[USER\]',            # Remove [USER] tags
        r'\[ASSISTANT\]',       # Remove [ASSISTANT] tags
        r'\[/.*?\]',            # Remove any [/TAG] patterns
        r'\[.*?\]',             # Remove any [TAG] patterns
        r'<.*?>',               # Remove HTML-like tags
        r'```.*?```',           # Remove code blocks
        r'`.*?`',               # Remove inline code
    ]
    
    cleaned = answer
    for pattern in patterns_to_remove:
        cleaned = re.sub(pattern, '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    
    # Clean up extra whitespace
    cleaned = re.sub(r'\n\s*\n', '\n\n', cleaned)  # Multiple newlines to double newlines
    cleaned = re.sub(r'^\s+|\s+$', '', cleaned)    # Trim leading/trailing whitespace
    cleaned = re.sub(r' +', ' ', cleaned)          # Multiple spaces to single space
    
    return cleaned


def process_question_with_agent(agent, question):
    """
    Process a user question using the RAG agent and return the answer and retrieved documents.
    
    Args:
        agent: The configured RAG agent
        question (str): The user's question
        
    Returns:
        tuple: (answer, retrieved_docs) where answer is the generated response 
               and retrieved_docs are the documents used for context
    """
    # Create the message for the agent
    messages = [
        HumanMessage(
            content=question
        )
    ]
    
    # Measure execution time
    start_time = time.time()
    
    # Invoke the agent to get the answer
    result = agent.invoke({"messages": messages})
    answer = result["messages"][-1].content if result.get("messages") else "No answer generated"
    
    # Clean the answer from unwanted formatting tags
    answer = clean_answer(answer)
    
    execution_time = time.time() - start_time
    
    # Add timing info to the answer for debugging
    answer_with_timing = f"{answer}\n\n*Response time: {execution_time:.2f} seconds*"
    
    return answer_with_timing


def get_retrieved_documents(retriever, question):
    """
    Get the documents retrieved for a given question.
    
    Args:
        retriever: The document retriever
        question (str): The user's question
        
    Returns:
        list: List of retrieved documents
    """
    return retrieve_documents(question, retriever)
