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
# QUESTION_ANALYSIS_CONFIG removed - analyze_question now uses main_model (gemma2:2b)


# ---------- PYDANTIC MODELS ----------
class QuestionAnalysis(BaseModel):
    """Model for question analysis and reformulation output."""
    reformulated_question: str = Field(description="Reformulated and clarified question")
    is_legal: bool = Field(description="Whether the question concerns law/legal matters")
    scope: str = Field(description="Legal scope/domain if applicable (e.g., 'fair markets', 'digital services', 'defective products')")


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


# ---------- ANALYSIS PROMPT ----------
def create_analysis_prompt():
    """Create prompt for question analysis and reformulation."""
    system_template = """
You are an expert legal classifier and document retrieval specialist.

**Your tasks:**
1. Reformulate the question for optimal document retrieval
2. Determine if the question is of a legal nature
3. Identify the specific legal scope/domain if applicable

**Question Reformulation Guidelines:**
Transform any question into a version that is:
1. Clear and grammatically correct
2. Complete, including any implicit context needed to understand the query fully
3. Specific, avoiding vague terms and including relevant keywords for retrieval
4. Focused on the core topic of the question
5. Concise but rich enough to maximize relevant document matches

**Legal Classification Criteria:**
A question is considered LEGAL (is_legal: true) if it involves:
- Laws, regulations, statutes, or legal frameworks
- Rights and obligations (individual, corporate, governmental)
- Contracts, agreements, or legal clauses
- Legal procedures, compliance, or litigation
- Intellectual property, labor, corporate, or commercial law
- Data protection, privacy, or cybersecurity regulations
- AI regulations, digital services, or technology law
- Any aspect where legal advice, references, or interpretations would be relevant

A question is considered NON-LEGAL (is_legal: false) if it involves:
- Pure technical questions (e.g., "How to install software?")
- General knowledge questions (e.g., "What is the capital of France?")
- Personal opinions or preferences
- Questions unrelated to legal matters

**Response Format:**
Respond ONLY with valid JSON containing:
1. "reformulated_question": "optimized question for document retrieval with clear context and relevant keywords"
2. "is_legal": true or false (based on criteria above)
3. "scope": "legal domain" (e.g., "data protection law", "AI regulation", "labor law", "contract law") if legal, otherwise "general"

Example 1 (Legal):
{{
  "reformulated_question": "What are the specific legal obligations and compliance requirements for AI systems under EU AI Act and GDPR regulations?",
  "is_legal": true,
  "scope": "AI regulation"
}}

Example 2 (Non-legal):
{{
  "reformulated_question": "How do I install and configure Python programming language on my computer?",
  "is_legal": false,
  "scope": "general"
}}

Response:"""

    return ChatPromptTemplate.from_messages([
        ("system", system_template),
        ("human", "{question}")
    ])


# ---------- PROMPT STRICT JSON ----------
def create_prompt_strict(scope: str | None = None):
    scope_text = " (Scope: " + str(scope) + ")" if scope else ""

    system_template = """
You are a legal assistant specialized in French and European law.
Follow these rules strictly:

**Document Retrieval**:
   - Always use the tool_rag to search for relevant legal documents
   - Include the legal scope/domain in your query for better document retrieval""" + scope_text + """
   - If a scope is provided above, incorporate it into your search query to find more relevant documents
   - Respond **only** with a JSON tool call in the following format,
     with no additional text:
{{
  "name": "tool_rag",
  "arguments": {{
    "query": "YOUR ACTUAL REFORMULATED QUESTION HERE"
  }}
}}
   - Replace "YOUR ACTUAL REFORMULATED QUESTION HERE" with the actual reformulated legal question
   - Do not include the placeholder text, use the real question instead
   - Example: If user asks "AI rules?", respond with: {{"name": "tool_rag", "arguments": {{"query": "What are the legal regulations and compliance requirements for artificial intelligence systems under EU law?"}}}}
   - Do not add any explanations, comments, or extra text outside this JSON.
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


def call_model(state: MessagesState, chat_model):
    if chat_model is None:
        raise ValueError("Chat model missing")

    messages = state.get("messages", [])

    # Extract scope from the last message if it has analysis data
    scope = None
    if messages:
        last_message = messages[-1]
        if hasattr(last_message, 'additional_kwargs') and last_message.additional_kwargs:
            scope = last_message.additional_kwargs.get("scope", None)

    try:
        # Use structured output for robust tool call parsing
        structured_model = chat_model.with_structured_output(ToolCallSchema)
        prompt = create_prompt_strict(scope=scope)
        structured_chain = prompt | structured_model

        # Single call with structured output
        tool_call_result = structured_chain.invoke({"history": messages})

        # Convert to tool_calls format expected by LangGraph
        tool_calls = [{
            "name": tool_call_result.name,
            "args": tool_call_result.arguments,
            "id": "call_1"
        }]

        print("[OK] Tool call parsed successfully with structured output")

        response = AIMessage(content="", tool_calls=tool_calls)

    except Exception as e:
        print(f"[ERROR] Structured tool call failed: {e}")
        # Fallback to original method
        prompt = create_prompt_strict(scope=scope)
        chat_model_with_prompt = prompt | chat_model
        response = chat_model_with_prompt.invoke({"history": messages})
        raw_content = response.content if hasattr(response, "content") else str(response)
        tool_calls = parse_tool_call(raw_content)

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

    # Detect language for the final answer to match user's original question
    try:
        lang_detected = detect(user_question)
    except Exception:
        lang_detected = None

    # Calculate dynamic response length limit using actual content
    max_response_tokens = calculate_max_response_tokens(latest_tool_message.content, user_question)

    # Create prompt and generate answer
    final_prompt = _create_final_prompt(user_question, latest_tool_message.content, lang_detected, max_response_tokens)

    try:
        response = chat_model.invoke(final_prompt)
        final_answer = response.content if hasattr(response, 'content') else str(response)
        return {"messages": [AIMessage(content=final_answer)]}
    except Exception as e:
        print(f"[ERROR] Failed to generate final answer: {e}")
        return {"messages": [AIMessage(content="I apologize, but I encountered an error while generating the final answer.")]}


# ---------- QUESTION ANALYSIS NODE ----------
def analyze_question(state: dict, question_model):
    """Analyze and reformulate the user question, and check if it's legal-related."""
    messages = state.get("messages", [])
    if not messages:
        return {"messages": [AIMessage(content="No question found to analyze.")]}

    # Get the latest user message
    user_message = None
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            user_message = msg
            break

    if not user_message:
        return {"messages": [AIMessage(content="No user question found to analyze.")]}

    original_question = user_message.content

    try:
        # Use structured output for robust JSON parsing
        structured_model = question_model.with_structured_output(QuestionAnalysis)
        prompt = create_analysis_prompt()
        structured_chain = prompt | structured_model

        # Get structured analysis
        analysis_result = structured_chain.invoke({"question": original_question})

        # Create analysis result message with metadata
        analysis_message = AIMessage(
            content=analysis_result.reformulated_question,
            additional_kwargs={
                "is_legal": analysis_result.is_legal,
                "reformulated_question": analysis_result.reformulated_question,
                "scope": analysis_result.scope
            }
        )
        return {
            "messages": [analysis_message]
        }

    except Exception as e:
        print(f"[ERROR] Question analysis failed: {e}")
        error_message = AIMessage(
            content=original_question,
            additional_kwargs={
                "is_legal": True,  # Default to legal for safety
                "reformulated_question": original_question,
                "scope": "general"
            }
        )
        return {
            "messages": [error_message]
        }


# ---------- ROUTING FUNCTION ----------
def route_after_analysis(state: dict):
    """Route to legal agent if question is legal, otherwise provide general response."""
    messages = state.get("messages", [])

    # Extract analysis data from the last message
    if messages:
        last_message = messages[-1]
        if hasattr(last_message, 'additional_kwargs') and last_message.additional_kwargs:
            is_legal = last_message.additional_kwargs.get("is_legal", True)
        else:
            is_legal = True  # Default to legal for safety
    else:
        is_legal = True  # Default to legal for safety

    if is_legal:
        return "agent"  # Route to the main legal agent
    else:
        return "general_response"  # Route to general response


# ---------- GENERAL RESPONSE NODE ----------
def provide_general_response(state: dict):
    """Provide a general response for non-legal questions."""
    messages = state.get("messages", [])

    # Extract reformulated question from the last message
    reformulated_question = ""
    if messages:
        last_message = messages[-1]
        if hasattr(last_message, 'additional_kwargs') and last_message.additional_kwargs:
            reformulated_question = last_message.additional_kwargs.get("reformulated_question", "")
        else:
            reformulated_question = last_message.content if hasattr(last_message, 'content') else ""

    response_content = f"""I am a legal assistant specialized in French and European law.

Your question: "{reformulated_question}"

This question does not appear to be directly related to the legal domain. I can help you with:
- Questions about French and European law
- Analysis of legal texts
- Clarification of legal regulations
- Advice on legal obligations

If you have a legal question, please feel free to reformulate it or ask me a new question."""

    return {"messages": [AIMessage(content=response_content)]}


# ---------- MAIN BUILDER ----------
def create_rag_agent(main_model, retriever, question_model=None, tool_model=None):
    """
    Create RAG agent with specialized model usage:
    - main_model (gemma2:2b): Question analysis, reformulation, and final answer generation
    - tool_model (gemma3:270m): Tool calls and document retrieval queries
    - question_model: Legacy parameter for compatibility (unused, analyze_question uses main_model)
    """
    tools = [create_rag_tool(retriever)]
    tool_node = BasicToolNode(tools)

    # Use tool_model for tool calls, fallback to question_model, then main_model
    effective_tool_model = tool_model or question_model or main_model

    workflow = StateGraph(MessagesState)

    # Add nodes with specialized model usage
    workflow.add_node("question_analysis", lambda state: analyze_question(state, main_model))  # gemma2:2b
    workflow.add_node("agent", lambda state: call_model(state, effective_tool_model))      # gemma3:270m
    workflow.add_node("tools", tool_node)                                                        # No model needed
    workflow.add_node("final_answer", lambda state: create_final_answer(state, main_model))      # gemma2:2b
    workflow.add_node("general_response", provide_general_response)                              # No model needed

    # Add edges
    workflow.add_edge(START, "question_analysis")

    # Route after question analysis
    workflow.add_conditional_edges("question_analysis", route_after_analysis, {
        "agent": "agent",
        "general_response": "general_response"
    })

    # Main agent flow
    workflow.add_conditional_edges("agent", route_tools, {
        "tools": "tools",
        END: END
    })
    workflow.add_edge("tools", "final_answer")
    workflow.add_edge("final_answer", END)

    # General response flow
    workflow.add_edge("general_response", END)

    return workflow.compile()
