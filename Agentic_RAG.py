import os
from dotenv import load_dotenv
from langchain_deepseek import ChatDeepSeek
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.tools import tool
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing import Literal
from langchain.messages import HumanMessage, AIMessage
from tavily import TavilyClient
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from IPython.display import Image, display



load_dotenv()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY")

llm_2 = ChatDeepSeek(
    model="deepseek-chat",
    temperature=0.6,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key= DEEPSEEK_API_KEY
)

class AgentState(MessagesState):
    retry_count: int
    intent: str | None

class QueryIntent(BaseModel):
    intent: Literal["greeting", "lds_religion", "web_search"]


embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
retriever = FAISS.load_local("E://Practice//Fastapi//RAG_FastApi//faiss_index", embedding_model,allow_dangerous_deserialization=True).as_retriever()

INTENT_PROMPT = """
You are classifying a user query into one of the following intents:

1. greeting – casual greetings or small talk
2. lds_religion – questions about LDS doctrine, teachings, or beliefs
3. web_search – general factual questions, recent events, or non-LDS topics

Respond with exactly one intent.
Query: {question}
"""

GRADE_PROMPT = (
    "You are a grader assessing relevance of a retrieved document to a user question. \n "
    "Here is the retrieved document: \n\n {context} \n\n"
    "Here is the user question: {question} \n"
    "If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. \n"
    "Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."
)

REWRITE_PROMPT = (
    "Look at the input and try to reason about the underlying semantic intent / meaning.\n"
    "Here is the initial question:"
    "\n ------- \n"
    "{question}"
    "\n ------- \n"
    "Formulate an improved question:"
)

GENERATE_PROMPT = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer the question. "
    "If you don't know the answer, just say that you don't know. "
    "Use three sentences maximum and keep the answer concise.\n"
    "Question: {question} \n"
    "Context: {context}"
)


# Tools
@tool
def retriever_tool(query: str) -> str:
    """Search and return information about LDS and it's teachings"""
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])


tavily_client = TavilyClient(api_key="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")

@tool(description="Do a web search based on user queries")
def web_search(query: str) -> str:
    result = tavily_client.search(query)
    return "\n\n".join(
        r["content"] for r in result.get("results", [])
    )


# Nodes
def route_initial_intent(state: AgentState):
    question = state["messages"][0].content

    response = llm_2.with_structured_output(QueryIntent).invoke(
        [{"role": "user", "content": INTENT_PROMPT.format(question=question)}]
    )

    return {"intent": response.intent}

def greeting_response(state: AgentState):
    response = llm_2.invoke(state["messages"])
    return {"messages": [response]}


def web_entry(state: AgentState):
    query = state["messages"][0].content
    result = tavily_client.search(query)

    content = "\n\n".join(r["content"] for r in result["results"])

    return {"messages": [HumanMessage(content=content)]}

def lds_retriever_node(state: AgentState):
    last_user_msg = next(
        m for m in reversed(state["messages"]) if m.type == "human"
    )
    docs_text = retriever_tool.invoke(last_user_msg.content)
    return {"messages": [AIMessage(content=docs_text)]}


class GradeDocuments(BaseModel):  
    """Grade documents using a binary score for relevance check."""

    binary_score: str = Field(
        description="Relevance score: 'yes' if relevant, or 'no' if not relevant"
    )


def grade_documents(
    state: AgentState,
) -> Literal["generate_answer", "rewrite_question", "web_fallback"]:

    question = state["messages"][0].content
    context = state["messages"][-1].content

    prompt = GRADE_PROMPT.format(question=question, context=context)
    response = llm_2.with_structured_output(GradeDocuments).invoke(
        [{"role": "user", "content": prompt}]
    )

    if response.binary_score == "yes":
        return "generate_answer"

    if state["retry_count"] >= 1:
        return "web_fallback"

    return "rewrite_question"



def rewrite_question(state: AgentState):
    """Rewrite the original user question."""
    messages = state["messages"]
    question = messages[0].content
    prompt = REWRITE_PROMPT.format(question=question)
    response = llm_2.invoke([{"role": "user", "content": prompt}])
    return {
        "messages": [HumanMessage(content=response.content)],
        "retry_count": state["retry_count"] + 1
            }


def decide_web_or_respond(state: AgentState):
    last_user_msg = next(
        m for m in reversed(state["messages"])
        if m.type == "human"
    )

    response = (
        llm_2
        .bind_tools([web_search])
        .invoke([last_user_msg])
    )

    return {"messages": [response]}


def web_fallback(state: AgentState):
    query = state["messages"][0].content
    result = web_search.invoke(query)

    return {
        "messages": [
            HumanMessage(content=result)
        ]
    }


def generate_answer(state: AgentState):
    """Generate an answer."""
    question = state["messages"][0].content
    context = state["messages"][-1].content
    prompt = GENERATE_PROMPT.format(question=question, context=context)
    response = llm_2.invoke([{"role": "user", "content": prompt}])
    return {"messages": [response]}


# Graph
workflow = StateGraph(AgentState)

workflow.add_node("route_intent", route_initial_intent)
workflow.add_node("greeting", greeting_response)
workflow.add_node("web_entry", web_entry)
workflow.add_node("lds_retriever_node",lds_retriever_node)
workflow.add_node("rewrite_question",rewrite_question)
workflow.add_node("generate_answer",generate_answer)
workflow.add_node("decide_web", decide_web_or_respond)
workflow.add_node("web_search", ToolNode([web_search]))


workflow.add_edge(START, "route_intent")

workflow.add_conditional_edges(
    "route_intent",
    lambda state: state["intent"],
    {
        "greeting": "greeting",
        "lds_religion": "lds_retriever_node",
        "web_search": "web_entry",
    }
)


workflow.add_conditional_edges(
    "lds_retriever_node",
    grade_documents,
    {
        "generate_answer": "generate_answer",
        "rewrite_question": "rewrite_question",
        "web_fallback": "decide_web",
    }
)

workflow.add_conditional_edges(
    "decide_web",
    tools_condition,
    {
        "tools": "web_search",
        END: END,
    }
)

workflow.add_edge("web_search", "generate_answer")
workflow.add_edge("generate_answer", END)
workflow.add_edge("rewrite_question", "lds_retriever_node")

# Compile
graph = workflow.compile()


# Display
png_bytes = graph.get_graph().draw_mermaid_png()

display(Image(png_bytes))
with open("rag_workflow.png", "wb") as f:
    f.write(graph.get_graph().draw_mermaid_png())


# Sample test
for chunk in graph.stream(
    {
        "messages": [
            {
                "role": "user",
                "content": "Who won icc t20 world cup 2021",
            }
        ],
        "retry_count": 0,
        "intent": ""
    }
):
    for node, update in chunk.items():
        print("Update from node", node)
        if "messages" in update:
            update["messages"][-1].pretty_print()
        else:
            print(update)
        print("\n\n")