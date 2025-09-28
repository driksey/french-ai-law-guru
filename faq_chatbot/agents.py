# agents.py
from langchain_core.messages import ToolMessage
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import START, END, MessagesState, StateGraph
from langchain_huggingface import ChatHuggingFace
from langchain.agents import Tool
from faq_chatbot.utils import retrieve_documents


class BasicToolNode:
    """Node for executing tools in the agent workflow."""
    
    def __init__(self, tools: list) -> None:
        # Outils disponibles
        self.tools_by_name = {tool.name: tool for tool in tools}

    def __call__(self, inputs: dict):
        # Récupère le dernier message de la liste "messages" dans les entrées
        if messages := inputs.get("messages", []):
            message = messages[-1]
        else:
            raise ValueError("No message found in input")
        outputs = []
        for tool_call in message.tool_calls:
            # Exécute l'outil spécifié et retourne le résultat
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
    """Create the system prompt for the agent."""
    return ChatPromptTemplate.from_messages([
        ("system", "Use the given RAG tool to answer the question. "
                    "If you don't know the answer, say you don't know. "
                    "Use three sentences maximum and keep the answer concise. "),
        ("placeholder", "{history}"),
    ])


def call_model(state: MessagesState, chat_model, tools):
    """Call the model with tools bound."""
    prompt = create_prompt()
    chat_model_with_tools = chat_model.bind_tools(tools)
    chat_model_with_prompt = prompt | chat_model_with_tools
    response = chat_model_with_prompt.invoke({"history": state["messages"]})
    return {"messages": response}


def route_tools(state: MessagesState):
    """Route to tools if tool calls are present, otherwise end."""
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and ai_message.tool_calls and len(ai_message.tool_calls) > 0:
        return "tools"
    return END


def create_rag_tool(retriever):
    """Create a RAG tool for document retrieval."""
    return Tool(
        name="tool_rag",
        description="""Tool to retrieve the k closest documents answering a question on IA regulation.""",
        func=lambda query: retrieve_documents(query, retriever)
    )


def create_rag_agent(chat_model, retriever):
    """Create a complete RAG agent with the document retrieval tool."""
    tool_rag = create_rag_tool(retriever)
    return define_graph(chat_model, [tool_rag])


def define_graph(chat_model, tools):
    """Define the agent workflow graph."""
    workflow = StateGraph(MessagesState)
    
    # Créer une fonction partielle pour call_model avec les outils
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
