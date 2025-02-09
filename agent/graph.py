import asyncio
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import START, END, StateGraph
from langchain_core.messages import  HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from .prompts import QUERY_WRITER_PROMPT, INFO_PROMPT, ENRICHMENT_PROMPT ,WEB_SEARCH_ROUTER_PROMPT, GET_USER_INPUT
from tavily import AsyncTavilyClient
from langgraph.checkpoint.mongodb.aio import AsyncMongoDBSaver
from pymongo import MongoClient,AsyncMongoClient
from .state import AgentState,UserInput,Queries,EnrichmentReport,WebSearchRouter
from .utils import format_search_results
import os, getpass
import graphviz


def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

_set_env("OPENAI_API_KEY")
_set_env("LANGCHAIN_API_KEY")
_set_env("TAVILY_API_KEY")
_set_env("MONGODB_STRING")

llm = ChatOpenAI(model="gpt-4o", temperature=0) 



def get_user_input(state: AgentState):
    """
    Extracts structured user input from raw input messages.
    
    Returns:
        dict: Contains 'company' and 'user_focus_topics' extracted from user input.
    """
    raw_user_inputs = state.raw_user_inputs
    structured_llm = llm.with_structured_output(UserInput)
    result = structured_llm.invoke([SystemMessage(content=GET_USER_INPUT)] + raw_user_inputs)
    return {"company" : result.company,
            "user_focus_topics" : result.user_focus_topics}

def generate_queries(state: AgentState):
    """
    Generates search queries based on user input and predefined structure.
    
    Returns:
        dict: A dictionary containing a list of generated search queries.
    """
    max_search_queries = 3
    structured_llm = llm.with_structured_output(Queries)
    query_instructions = QUERY_WRITER_PROMPT.format(
        company=state.company,
        user_focus_topics=state.user_focus_topics,
        max_search_queries=max_search_queries,
    )
    # Generate queries using the language model
    results = structured_llm.invoke([SystemMessage(content=query_instructions)] + [HumanMessage(content="Generate a list of querys")])

    query_list = [query for query in results.queries]
    return {"search_queries": query_list}

async def search_web(state: AgentState):    
    """
    Performs web searches asynchronously using Tavily API.

    Returns:
        dict: A dictionary containing search results.
    """
    tavily_search = AsyncTavilyClient()
    search_tasks = []

    # Determine whether to use initial queries or enrichment queries
    if(not state.enrichment_queries):
        online_search_queries = state.search_queries
    else:
        online_search_queries = state.enrichment_queries

    # Create async tasks for each query
    for query in online_search_queries:
            search_tasks.append(tavily_search.search(query,max_results=2))
    
    # Execute all search tasks concurrently
    search_results = await asyncio.gather(*search_tasks) 
    return{"search_results" : search_results}


def write_report(state: AgentState):
    """
    Generates a final report based on search results.

    Returns:
        dict: A dictionary containing the final report content.
    """
    search_results = state.search_results
    search_results_str = format_search_results(search_results)

    answer_prompt = INFO_PROMPT.format(
        company=state.company,
        queries=state.search_queries,
        enrichment_queries=state.enrichment_queries,
        user_focus_topics=state.user_focus_topics,
        content=search_results_str
    )

    report = llm.invoke([SystemMessage(content=answer_prompt)]) 

    return{"final_report" : [report.content]}


def Enrichment(state: AgentState):
    """
    Determines whether additional enrichment is required based on the generated report.

    Returns:
        dict: Contains enrichment queries or signals completion.
    """
    structured_llm = llm.with_structured_output(EnrichmentReport)

    enrichment_prompt = ENRICHMENT_PROMPT.format(
        queries=state.search_queries,
        enrichment_queries=state.enrichment_queries,
        report=state.final_report[-1]
    )

    result = structured_llm.invoke([SystemMessage(content=enrichment_prompt)] + [HumanMessage(content="Produce a structured Enrichment report.")])

    if result.is_satisfied:
        return {"is_satisfied" : result.is_satisfied}
    else:
        return {"is_satisfied" : result.is_satisfied,
                "enrichment_steps_taken" : state.enrichment_steps_taken + 1 ,
                "enrichment_queries": result.enrichment_queries}

def router(state: AgentState):
    """
    Determines whether the agent should continue enrichment or stop.

    Returns:
        str: search_web node if enrichment is needed, else END
    """
    max_enrichment_steps = 3
    enrichment_steps_taken = state.enrichment_steps_taken

    if state.is_satisfied or enrichment_steps_taken >= max_enrichment_steps:
        state.is_satisfied = False
        state.enrichment_steps_taken = 0
        state.enrichment_queries = []
        return END
    
    return "search_web"

def web_search_router(state: AgentState):
    """
    Determines whether an additional online search is needed or if the existing data is sufficient.

    Returns:
        str: routs to search_web node if online search is needed, else routs to write_reporte node
    """
    structured_llm = llm.with_structured_output(WebSearchRouter) 
    
    web_search_router_prompt = WEB_SEARCH_ROUTER_PROMPT.format(
        queries=state.search_queries,
        results=state.search_results
    )
    result = structured_llm.invoke([SystemMessage(content=web_search_router_prompt)])
    if(result.is_search_results_answers_queries):
        return "write_report"
    return "search_web"

def build_agent():
    builder = StateGraph(AgentState)

    # Define  nodes
    builder.add_node("get_user_input",get_user_input)
    builder.add_node("generate_queries" , generate_queries)
    builder.add_node("search_web", search_web)
    builder.add_node("write_report", write_report)
    builder.add_node("Enrichment", Enrichment)

    # Define edges
    builder.add_edge(START, "get_user_input")
    builder.add_edge("get_user_input","generate_queries")
    builder.add_conditional_edges("generate_queries",web_search_router)
    builder.add_edge("search_web", "write_report")
    builder.add_edge("write_report" , "Enrichment")
    builder.add_conditional_edges("Enrichment",router)

    # Initialize MongoDB checkpointer
    MONGODB_URI = os.environ.get('MONGODB_STRING')
    mongodb_client = AsyncMongoClient(MONGODB_URI)
    checkpointer = AsyncMongoDBSaver(mongodb_client)

    graph = builder.compile(checkpointer=checkpointer)

    return graph

