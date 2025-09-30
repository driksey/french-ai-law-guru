# agents.py

import json
import re
from typing import List
from langdetect import detect

from langchain.agents import Tool
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

from legal_ai_assistant.utils import (
    retrieve_documents,
    calculate_max_response_tokens
)


# ---------- SCHEMA ----------
class ToolCallSchema(BaseModel):
    name: str = Field(description="Function name")
    arguments: dict = Field(description="Function arguments")


# ---------- TOOL NODE ----------
class BasicToolNode:
    def __init__(self, tools: List[Tool]) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        messages = inputs.get("messages", [])
        if not messages:
            raise ValueError("No messages found")

        message = messages[-1]
        tool_calls = getattr(message, "tool_calls", [])

        outputs = []
        for tool_call in tool_calls:
            tool_name, tool_args = tool_call["name"], tool_call["args"]
            if tool_name not in self.tools_by_name:
                raise ValueError(f"Unknown tool requested: {tool_name}")

            try:
                tool_result = self.tools_by_name[tool_name].invoke(tool_args)
                outputs.append(
                    ToolMessage(
                        content=str(tool_result),
                        name=tool_name,
                        tool_call_id=tool_call.get("id", "parsed-tool-call"),
                    )
                )
            except Exception as e:
                outputs.append(
                    ToolMessage(
                        content=f"[ERROR] Tool {tool_name} failed: {e}",
                        name=tool_name,
                        tool_call_id=tool_call.get("id", "parsed-tool-call"),
                    )
                )

        return {"messages": outputs}


# ---------- RAG TOOL ----------
def create_rag_tool(retriever):
    return Tool(
        name="tool_rag",
        description="Retrieve legal documents (EU AI Act, French AI law).",
        func=lambda query: retrieve_documents(query, retriever)
    )


# ---------- PROMPT STRICT JSON ----------
def create_prompt_strict(language_hint: str | None = None):
    language_text = language_hint if language_hint else "detect automatically"
    
    system_template = """
You are an assistant specialized in analyzing user questions.
Follow these rules strictly:

1. Language: Always respond in the same language as the user's question (""" + language_text + """).

2. **Legal Question Detection**:
   - If the question is legal in nature (laws, rights, contracts, regulations,
     case law, legal obligations, etc.):
     - Reformulate the question to make it clearer, more detailed, and precise
       for document retrieval.
     - Respond **only** with a JSON tool call in the following format,
       with no additional text:  
{{
  "name": "tool_rag",
  "arguments": {{
    "query": "<reformulated complete legal question with context>"
  }}
}}
   - Do not add any explanations, comments, or extra text outside this JSON.  

3. **Non-Legal Questions**:  
   - If the question is not legal in nature, respond normally and directly, without any tool call or JSON.  
"""
    
    return ChatPromptTemplate.from_messages([
        ("system", system_template),
        MessagesPlaceholder("history")
    ])


# ---------- MODEL CALL ----------
def parse_tool_call(raw_content: str):
    """Robust JSON parsing using regex for better reliability."""
    candidates = []
    
    # Remove markdown code blocks if present
    clean_content = raw_content.strip()
    if clean_content.startswith('```json'):
        clean_content = clean_content.replace('```json', '').replace('```', '').strip()
    elif clean_content.startswith('```'):
        clean_content = clean_content.replace('```', '').strip()
    
    # Use regex to find JSON blocks between { and }
    json_pattern = r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}'
    json_blocks = re.findall(json_pattern, clean_content, re.DOTALL)
    
    # If no blocks found, try the whole content
    if not json_blocks:
        json_blocks = [clean_content]
    
    for block in json_blocks:
        try:
            parsed = json.loads(block.strip())
            validated = ToolCallSchema(**parsed)
            candidates.append({
                "id": "parsed-tool-call",
                "name": validated.name,
                "args": validated.arguments
            })
            break  # Return first valid JSON found
        except Exception:
            continue

    return candidates


def call_model(state: MessagesState, chat_model, tools):
    if chat_model is None:
        raise ValueError("Chat model missing")

    # Detect language of last user message
    user_messages = [m for m in state["messages"] if isinstance(m, HumanMessage)]
    last_user_msg = user_messages[-1].content if user_messages else "Hello"
    try:
        lang_detected = detect(last_user_msg)
    except Exception:
        lang_detected = None

    prompt = create_prompt_strict(language_hint=lang_detected)
    chat_model_with_prompt = prompt | chat_model

    # Single call without retry mechanism
    response = chat_model_with_prompt.invoke({"history": state["messages"]})
    raw_content = response.content if hasattr(response, "content") else str(response)
    tool_calls = parse_tool_call(raw_content)
    
    if tool_calls:
        print("[OK] Tool call parsed successfully")

    if isinstance(response, AIMessage):
        response.tool_calls = tool_calls
    else:
        response = AIMessage(content=raw_content, tool_calls=tool_calls)

    # Minimal logging for speed
    return {"messages": [response]}


# ---------- ROUTING ----------
def route_tools(state: MessagesState):
    """Route based on message types for robust workflow."""
    messages = state if isinstance(state, list) else state.get("messages", [])
    if not messages:
        raise ValueError("No messages in route_tools")

    ai_message = messages[-1]
    
    # If AI message has tool calls, go to tools
    if getattr(ai_message, "tool_calls", []):
        return "tools"
    
    # If we have tool messages in history, go to final answer
    tool_messages = [msg for msg in messages if hasattr(msg, 'tool_call_id')]
    if tool_messages:
        return "final_answer"

    # If no tool calls, end the conversation
    return END


# ---------- FINAL ANSWER ----------

def _extract_messages_from_state(state: MessagesState):
    """Extract messages from state, handling both list and dict formats."""
    if isinstance(state, list):
        return state
    elif messages := state.get("messages", []):
        return messages
    else:
        raise ValueError(f"No messages found in input state: {state}")


def _find_user_question(messages):
    """Find the original user question from messages."""
    for message in messages:
        if isinstance(message, HumanMessage):
            return message.content
    return "the user's question"


def _create_final_prompt(user_question, tool_content, lang_detected, max_tokens):
    """Create the final answer prompt."""
    return f"""You are a legal assistant.
Question (language={lang_detected}): {user_question}

Documents:
{tool_content}

Answer with:
- Respond in the SAME LANGUAGE as the question
- Concise but complete explanation
- Cite exact legal references (e.g., "Article 5, EU AI Act 2024")
- No placeholders, only real citations
- Keep response under {max_tokens} tokens to avoid truncation
"""


def create_final_answer(state: MessagesState, chat_model):
    """Generate the final answer after tool execution - optimized for the new workflow."""
    messages = _extract_messages_from_state(state)

    # Extract tool results
    tool_messages = [msg for msg in messages if hasattr(msg, 'tool_call_id')]
    
    if not tool_messages:
        return {"messages": [AIMessage(content="No tool results found.")]}

    # Get the latest tool result
    latest_tool_message = tool_messages[-1]
    
    # Find the original user question
    user_question = _find_user_question(messages)
    
    # Detect language for the final answer
    try:
        lang_detected = detect(user_question)
    except Exception:
        lang_detected = None

    # Calculate dynamic response length limit
    doc_content_length = len(latest_tool_message.content)
    num_docs = len(tool_messages)
    max_response_tokens = calculate_max_response_tokens(doc_content_length, num_docs)
    
    # Create prompt and generate answer
    final_prompt = _create_final_prompt(user_question, latest_tool_message.content, 
                                       lang_detected, max_response_tokens)

    try:
        response = chat_model.invoke(final_prompt)
        final_answer = response.content if hasattr(response, 'content') else str(response)
        return {"messages": [AIMessage(content=final_answer)]}
    except Exception as e:
        print(f"[ERROR] Failed to generate final answer: {e}")
        return {"messages": [AIMessage(content="I apologize, but I encountered an error while generating the final answer.")]}
    

# ---------- MAIN BUILDER ----------
def create_rag_agent(chat_model, retriever):
    tools = [create_rag_tool(retriever)]
    tool_node = BasicToolNode(tools)

    workflow = StateGraph(MessagesState)
    workflow.add_node("agent", lambda state: call_model(state, chat_model, tools))
    workflow.add_node("tools", tool_node)
    workflow.add_node("final_answer", lambda state: create_final_answer(state, chat_model))

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", route_tools, {
        "tools": "tools",
        END: END
    })
    workflow.add_edge("tools", "final_answer")
    workflow.add_edge("final_answer", END)

    return workflow.compile()
