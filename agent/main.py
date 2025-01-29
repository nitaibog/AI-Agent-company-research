from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph.message import AnyMessage, add_messages
from utils.tools import search_web
from typing import Any, List, Optional
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from IPython.display import Image, display
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import Runnable
from langchain_openai import ChatOpenAI
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import MessagesState
from typing import Annotated, List
from utils.configuration import Configuration
from utils.prompts import QUERY_WRITER_PROMPT
import os, getpass


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
_set_env("LANGCHAIN_API_KEY")
_set_env("TAVILY_API_KEY")


llm = ChatOpenAI(model="gpt-4o", temperature=0) 
# tools = [search_web]
# llm_with_tools =  llm.bind_tools(tools, parallel_tool_calls=False)

# sys_msg = SystemMessage(content="You are a helpful assistant tasked with online search ")

class AgentState(MessagesState):
    company: str = Field(description="The name of the company")
    search_queries: list[str] = Field(
        description="List of search queries.",
    )
    pass


class Queries(BaseModel):
    queries: list[str] = Field(
        description="List of search queries.",
    )


def generate_queries(state: AgentState):
    """Generate search queries based on the user input and extraction schema."""
    # Get configuration
    # configurable = Configuration.from_runnable_config(config)
    max_search_queries = 6

    # Generate search queries
    structured_llm = llm.with_structured_output(Queries)

    # Format system instructions
    query_instructions = QUERY_WRITER_PROMPT.format(
        company=state["company"],
        user_notes=state["messages"],
        max_search_queries=max_search_queries,
    )
    # Generate queries
    results = structured_llm.invoke([SystemMessage(content=query_instructions)] + [HumanMessage(content="Generate a list of querys")])

    # Queries
    query_list = [query for query in results.queries]
    return {"search_queries": query_list}

builder = StateGraph(AgentState)

builder.add_node("generate_queries" , generate_queries)

builder.add_edge(START, "generate_queries")
builder.add_edge("generate_queries", END)

memory = MemorySaver()

graph = builder.compile(checkpointer=memory)

display(Image(graph.get_graph(xray=1).draw_mermaid_png()))


# def search_web(state: AgentState):
    
#     """ Retrieve docs from web search """
#     tavily_search = TavilySearchResults(max_results=3)
#     # Search query
#     structured_llm = llm.with_structured_output(SearchQuery)
#     search_query = structured_llm.invoke([search_instructions]+state['messages'])
    
#     # Search
#     search_docs = tavily_search.invoke(search_query.search_query)

#      # Format
#     formatted_search_docs = "\n\n---\n\n".join(
#         [
#             f'<Document href="{doc["url"]}"/>\n{doc["content"]}\n</Document>'
#             for doc in search_docs
#         ]
#     )

#     return {"context": [formatted_search_docs]} 

