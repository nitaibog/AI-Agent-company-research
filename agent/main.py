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
from utils.prompts import QUERY_WRITER_PROMPT, INFO_PROMPT, ENRICHMENT_PROMPT
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

class AgentState(BaseModel):
    raw_user_input: str = Field(
        description="The raw user input"
    )
    company: str = Field(default=None,
        description="The name of the company"
    )
    user_focus_topics : list[str] = Field(default=None,
        description="List of the focus topics for research on the company from the user's input"
    )
    search_queries: list[str] = Field(default=None,
        description="List of search queries.",
    )
    search_results:  list[dict] = Field(default=None,
        description="Results of the online search"
    ) #Annotated[list, operator.add] # Source docs
    report: Annotated[list, operator.add] = Field(default=None,
        description="The final report"
    )
    is_satisfied : bool = Field(default=None,
        description="true if the report is satisfactory"
    )
    enrichment_steps_taken : int = Field(default=0,
        description="Number of enrichment Steps taken"
    )
    
class UserInput(BaseModel):
    company: str = Field(
        description="The name of the company"
    )
    user_focus_topics : list[str] = Field(
        description="The focus topics for research on the company from the user's input"
    )

class Queries(BaseModel):
    queries: list[str] = Field(
        description="List of search queries."
    )

class EnrichmentReport(BaseModel):
    is_satisfied : bool = Field(
        description="true if the report is satisfactory"
    )
    search_queries : list[str] = Field(
        description="If is_satisfied is False, provide 1-3 targeted search queries to find the missing information and to enreach the data"
    )
    reasoning: str = Field(description="Brief explanation of the assessment")

def get_user_input(state: AgentState):
    # user_text = input("Enter company name and focus topics: ")
    # user_input = "hello, i want to get e report about Tesla. i want to focus about the companys stock whether i should invest in it"
    user_input = state.raw_user_input
    structured_llm = llm.with_structured_output(UserInput)
    result = structured_llm.invoke([HumanMessage(content=user_input)])
    return {"company" : result.company,
            "user_focus_topics" : result.user_focus_topics}

def generate_queries(state: AgentState):
    """Generate search queries based on the user input and extraction schema."""
    # Get configuration
    # configurable = Configuration.from_runnable_config(config)
    max_search_queries = 4

    # Generate search queries
    structured_llm = llm.with_structured_output(Queries)

    # Format system instructions
    query_instructions = QUERY_WRITER_PROMPT.format(
        company=state.company,
        user_notes=state.user_focus_topics,
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
        for query in state.search_queries:
             print(f"online search query: {query}")
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
    search_results = state.search_results
    search_results_str = format_search_results(search_results)
    answer_prompt = INFO_PROMPT.format(
        company=state.company,
        queries=state.search_queries,
        user_notes=state.user_focus_topics,
        content=search_results_str
    )
    report = llm.invoke([SystemMessage(content=answer_prompt)]) 
    return{"report" : [report.content]}

def Enrichment(state: AgentState):
    structured_llm = llm.with_structured_output(EnrichmentReport)
    enrichment_prompt = ENRICHMENT_PROMPT.format(
        queries=state.search_queries,
        report=state.report
    )
    result = structured_llm.invoke([SystemMessage(content=enrichment_prompt)] + [HumanMessage(content="Produce a structured Enrichment report.")])
    if result.is_satisfied:
        return {"is_satisfied" : result.is_satisfied}
    else:
        print(f"reasoning for enrichment step : {result.reasoning} /n/n")
        print(f"new queries {result.search_queries} /n/n")
        return {"is_satisfied" : result.is_satisfied,
                "enrichment_steps_taken" : state.enrichment_steps_taken + 1 ,
                "search_queries": result.search_queries}

def router(state: AgentState):
    max_enrichment_steps = 3
    enrichment_steps_taken = state.enrichment_steps_taken
    if state.is_satisfied:
        return END
    if enrichment_steps_taken >= max_enrichment_steps:
        return END
    return "search_web"

builder = StateGraph(AgentState)

builder.add_node("get_user_input",get_user_input)
builder.add_node("generate_queries" , generate_queries)
builder.add_node("search_web", search_web)
builder.add_node("write_report", write_report)
builder.add_node("Enrichment", Enrichment)

builder.add_edge(START, "get_user_input")
builder.add_edge("get_user_input","generate_queries")
builder.add_edge("generate_queries", "search_web")
builder.add_edge("search_web", "write_report")
builder.add_edge("write_report" , "Enrichment")
builder.add_conditional_edges("Enrichment",router)

memory = MemorySaver()

graph = builder.compile(checkpointer=memory)

# display(Image(graph.get_graph(xray=1).draw_mermaid_png()))


# company = "Tesla"
# thread = {"configurable": {"thread_id": "3"}}
async def main():
    company = "Tesla"
    thread = {"configurable": {"thread_id": "1"}}
    user_input = input("Enter company name and focus topics: ")
    async for event in graph.astream(
        # {"company": company, "user_notes": "please provide data about the company's stock and growth potential"}, 
        {"raw_user_input" : user_input},
        thread, 
        stream_mode="values"
    ):
        # Review search queries
        # search_queries = event.get("search_queries", "")
        # if search_queries:
        #     for query in search_queries:
        #         print(f"search_queries: {query}")
        #         print("-" * 50)
        report = event.get("report" , "")
        # if event.get("report" , ""):
        #     print(f"final report : {report}")
        print(f"num of enrichment steps : {event.get("enrichment_steps_taken")}")
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

