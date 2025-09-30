# agents.py

import json
import time
from typing import List
from langdetect import detect

from langchain.agents import Tool
from langchain_core.messages import ToolMessage, AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

from legal_ai_assistant.utils import retrieve_documents


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
def create_prompt_strict(language_hint: str = None):
    language_text = language_hint if language_hint else "detect automatically"
    
    system_template = """
You are a French AI Law Assistant specialized in EU AI Act and French AI regulations.

RULES:
- ALWAYS respond ONLY with a JSON tool call if the question is legal.
- DO NOT write any explanations or text outside the JSON.
- Respond in the SAME LANGUAGE as the user's question (""" + language_text + """).

JSON Schema Example:
{{
  "name": "tool_rag",
  "arguments": {{
    "query": "<the legal question>"
  }}
}}

If the question is not legal, answer normally in the same language.
"""
    
    return ChatPromptTemplate.from_messages([
        ("system", system_template),
        MessagesPlaceholder("history")
    ])


# ---------- MODEL CALL ----------
def parse_tool_call(raw_content: str):
    """Robust JSON parsing for multi-blocks."""
    candidates = []
    clean_content = raw_content.strip().replace("```json", "").replace("```", "").strip()
    possible_blocks = [clean_content] if "}}\n{{" not in clean_content else clean_content.split("\n")

    for block in possible_blocks:
        try:
            parsed = json.loads(block)
            validated = ToolCallSchema(**parsed)
            candidates.append({
                "id": "parsed-tool-call",
                "name": validated.name,
                "args": validated.arguments
            })
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
        print(f"[INFO] Detected user language: {lang_detected}")
    except Exception:
        lang_detected = None

    start_time = time.time()
    prompt = create_prompt_strict(language_hint=lang_detected)
    chat_model_with_prompt = prompt | chat_model

    # 🔹 Retry mechanism if JSON invalid
    max_retries = 2
    raw_content = ""
    tool_calls = []
    for attempt in range(max_retries):
        response = chat_model_with_prompt.invoke({"history": state["messages"]})
        raw_content = response.content if hasattr(response, "content") else str(response)
        tool_calls = parse_tool_call(raw_content)
        if tool_calls:
            print(f"[OK] Tool call parsed successfully on attempt {attempt+1}")
            break
        else:
            print(f"[WARN] JSON parse failed, retrying attempt {attempt+1}")
            # Add a retry instruction to force JSON only
            retry_msg = HumanMessage(content="Output valid JSON only. Do not write any text outside JSON.")
            state["messages"].append(retry_msg)

    if isinstance(response, AIMessage):
        response.tool_calls = tool_calls
    else:
        response = AIMessage(content=raw_content, tool_calls=tool_calls)

    print(f"[PERF] call_model total: {time.time() - start_time:.2f}s")
    return {"messages": [response]}


# ---------- ROUTING ----------
def route_tools(state: MessagesState):
    messages = state if isinstance(state, list) else state.get("messages", [])
    if not messages:
        raise ValueError("No messages in route_tools")

    ai_message = messages[-1]
    if getattr(ai_message, "tool_calls", []):
        return "tools"

    # If no tool calls, end the conversation
    return END


# ---------- FINAL ANSWER ----------
def create_final_answer(state: MessagesState, chat_model):
    """Generate the final answer after tool execution - optimized for the new workflow."""
    if isinstance(state, list):
        messages = state
    elif messages := state.get("messages", []):
        pass
    else:
        raise ValueError(f"No messages found in input state: {state}")

    # Extract tool results
    tool_messages = [msg for msg in messages if hasattr(msg, 'tool_call_id')]
    
    if not tool_messages:
        return {"messages": [AIMessage(content="No tool results found.")]}

    # Get the latest tool result
    latest_tool_message = tool_messages[-1]
    
    # Find the original user question using message type
    user_question = None
    for message in messages:
        if hasattr(message, 'type') and message.type == "human":
            user_question = message.content
            break
    
    if not user_question:
        user_question = "the user's question"
    
    # Detect language for the final answer
    try:
        lang_detected = detect(user_question)
        print(f"[INFO] Detected user language for final answer: {lang_detected}")
    except Exception:
        lang_detected = None

    # Create a streamlined prompt for faster generation
    final_prompt = f"""You are a legal assistant.
Question (language={lang_detected}): {user_question}

Documents:
{latest_tool_message.content}

Answer with:
- Respond in the SAME LANGUAGE as the question
- Concise but complete explanation
- Cite exact legal references (e.g., "Article 5, EU AI Act 2024")
- No placeholders, only real citations
"""

    try:
        # Generate final answer using the chat model with optimized settings
        import time
        final_start = time.time()
        response = chat_model.invoke(final_prompt)
        final_time = time.time() - final_start
        print(f"[PERF] Final answer generation time: {final_time:.2f}s")
        
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
