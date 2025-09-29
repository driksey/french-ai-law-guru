# agents.py

from langchain.agents import Tool
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import END, START, MessagesState, StateGraph
from pydantic import BaseModel, Field

from legal_ai_assistant.utils import retrieve_documents


class ToolCallSchema(BaseModel):
    """Schema for structured tool calling."""
    name: str = Field(description="The name of the function to call")
    arguments: dict = Field(description="The arguments for the function call")

    class Config:
        json_schema_extra = {
            "example": {
                "name": "tool_rag",
                "arguments": {
                    "query": "What are the main requirements for AI systems under the AI Act?"
                }
            }
        }


class BasicToolNode:
    """Node for executing tools in the agent workflow."""

    def __init__(self, tools: list) -> None:
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            tool_result = self.tools_by_name[tool_call["name"]].invoke(
                tool_call["args"]
            )
            outputs.append(
                ToolMessage(
                    content=str(tool_result),
                    name=tool_call["name"],
                    tool_call_id=tool_call["id"],
                )
            )
        return {"messages": outputs}


def create_prompt():
    """Create the system prompt optimized for multilingual responses."""
    return ChatPromptTemplate.from_messages([
        ("system", """AI Q&A assistant for French AI regulations. You MUST use tool_rag for ALL questions.

MANDATORY WORKFLOW:
1. ALWAYS call tool_rag with user question first
2. READ the documents returned by tool_rag
3. THEN provide answer based ONLY on those documents

TOOL: tool_rag(query) - Search AI regulation documents

FORMAT: {{"name": "tool_rag", "arguments": {{"query": "question"}}}}

AFTER TOOL CALL:
- Read the documents returned by tool_rag
- Use the information from these documents to answer the question
- Do NOT give generic responses like "Please call tool_rag"
- Provide specific answers based on the document content

ITERATIVE SEARCH (if needed):
- If the initial search doesn't provide complete information, make additional tool_rag calls
- Use different search terms to find complementary information
- Example: First search "GDPR requirements", then search "GDPR international transfers"
- Combine information from multiple searches for comprehensive answers

LANGUAGE RULE - CRITICAL: 
1. FIRST: Detect the language of the user's question
2. THEN: Respond in that EXACT language

Examples:
- French question "Quelles sont les exigences?" → French answer "Les exigences sont..."
- English question "What are the requirements?" → English answer "The requirements are..."
- Spanish question "¿Cuáles son los requisitos?" → Spanish answer "Los requisitos son..."

DETECT the language of the user's question and RESPOND in that EXACT language.

CRITICAL: You MUST use tool_rag before answering. Do not answer without searching documents first.

IMPORTANT: Keep responses concise and focused. Limit response to maximum 300 tokens."""),
        ("placeholder", "{history}"),
    ])


def call_model(state: MessagesState, chat_model, tools):
    """Call the model with tools bound."""
    if chat_model is None:
        raise ValueError("Chat model is None. Please check if the model loaded correctly.")

    prompt = create_prompt()
    chat_model_with_tools = chat_model.bind_tools(tools)
    chat_model_with_prompt = prompt | chat_model_with_tools
    response = chat_model_with_prompt.invoke({"history": state["messages"]})
    return {"messages": response}


def route_tools(state: MessagesState):
    """Route to tools if tool calls are present or if structured output is detected, otherwise end."""
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")

    print(f"[DEBUG] Routing decision for message: {type(ai_message).__name__}")

    # Check for traditional tool calls
    if hasattr(ai_message, "tool_calls") and ai_message.tool_calls and len(ai_message.tool_calls) > 0:
        print("[OK] Traditional tool calls detected")
        return "tools"

    print("[INFO] No tool calls detected, ending conversation")
    return END


def create_rag_tool(retriever):
    """Create a RAG tool for document retrieval with explicit schema."""
    return Tool(
        name="tool_rag",
        description="""ESSENTIAL tool to search for relevant information in AI regulation documents.

        Schema:
        - name: tool_rag
        - parameters: {"query": "string"}
        - description: Search for relevant information in AI regulation documents

        Usage: ALWAYS use this tool first before answering any question.
        Input: A question or search query about AI regulations in France.
        Output: Relevant document excerpts that help answer the question.

        Example: tool_rag(query="What are the AI Act requirements?")""",
        func=lambda query: retrieve_documents(query, retriever)
    )


def create_rag_agent(chat_model, retriever):
    """Create a complete RAG agent with the document retrieval tool."""
    tool_rag = create_rag_tool(retriever)
    return define_graph(chat_model, [tool_rag])


def define_graph(chat_model, tools):
    """Define the agent workflow graph with structured output."""
    workflow = StateGraph(MessagesState)

    def call_model_with_tools(state):
        return call_model(state, chat_model, tools)

    workflow.add_node("model", call_model_with_tools)

    tool_node = BasicToolNode(tools=tools)
    workflow.add_node("tools", tool_node)

    workflow.add_conditional_edges(
        "model",
        route_tools,
        {"tools": "tools", END: END},
    )

    workflow.add_edge(START, "model")
    workflow.add_edge("tools", "model")

    agent = workflow.compile()
    return agent
