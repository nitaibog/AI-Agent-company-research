import asyncio
import operator
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph.message import AnyMessage, add_messages
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
from utils.prompts import QUERY_WRITER_PROMPT, INFO_PROMPT
from tavily import AsyncTavilyClient
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
    search_results:  list[str] #Annotated[list, operator.add] # Source docs
    
    pass


class Queries(BaseModel):
    queries: list[str] = Field(
        description="List of search queries.",
    )

def generate_queries(state: AgentState):
    """Generate search queries based on the user input and extraction schema."""
    # Get configuration
    # configurable = Configuration.from_runnable_config(config)
    max_search_queries = 4

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

async def search_web(state: AgentState):
        tavily_search = AsyncTavilyClient()
        search_tasks = []
        for query in state["search_queries"]:
             search_tasks.append(tavily_search.search(query,max_results=3))
        search_results = await asyncio.gather(*search_tasks) #list[dict]
        return{"search_results" : search_results}

def format_search_results(search_results: list[dict] ):
    sources_list = []
    for result in search_results:
        sources_list.extend(result["results"])
    formatted_text = "Sources:\n\n"
    for source in sources_list:
        formatted_text += f"Source {source['title']}:\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += (
            f"Most relevant content from source: {source['content']}\n===\n"
        )
    return formatted_text



def write_report(state: AgentState):
    search_results = state["search_results"]
    search_results_str = format_search_results(search_results)
    answer_prompt = INFO_PROMPT.format(
        company=state["company"],
        queries=state["queries"],
        user_notes=state["messages"],
        content=search_results_str
    )
    report = llm.invoke([SystemMessage(content=answer_prompt)])
    print(report.content)
    return{"messages" : report}


builder = StateGraph(AgentState)

builder.add_node("generate_queries" , generate_queries)
builder.add_node("search_web", search_web)
builder.add_node("write_report", write_report)

builder.add_edge(START, "generate_queries")
builder.add_edge("generate_queries", "search_web")
builder.add_edge("search_web", "write_report")
builder.add_edge("write_report", END)

memory = MemorySaver()

graph = builder.compile(checkpointer=memory)

# display(Image(graph.get_graph(xray=1).draw_mermaid_png()))


# company = "Tesla"
# thread = {"configurable": {"thread_id": "3"}}
async def main():
    company = "Tesla"
    thread = {"configurable": {"thread_id": "5"}}

    async for event in graph.astream(
        {"company": company, "user_notes": "please provide data about the company's stock and growth potential"}, 
        thread, 
        stream_mode="values"
    ):
        # Review search queries
        search_queries = event.get("search_queries", "")
        if search_queries:
            for query in search_queries:
                print(f"search_queries: {query}")
                print("-" * 50)

        # # Review search results
        # search_results = event.get("search_results", "")
        # if search_results:
        #     for result in search_results:
        #         print(f"search_results: {result}")
        #         print("-" * 50)

# Run the async function
if __name__ == "__main__":
    asyncio.run(main())

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

