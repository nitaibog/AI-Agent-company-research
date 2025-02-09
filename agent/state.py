from pydantic import BaseModel, Field
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from typing import Annotated, List
import operator



class AgentState(BaseModel):
    raw_user_inputs: Annotated[list[AnyMessage], operator.add] = Field(
        description="The raw user inputs"
    )
    company: str = Field(default=None,
        description="The name of the company"
    )
    user_focus_topics : list[str] = Field(default=None,
        description="A list of topics to focus on if the user has specified such"
    )
    search_queries: list[str] = Field(default=None,
        description="List of search queries.",
    )
    enrichment_queries: list[str] = Field(default=None,
        description="List of enrichment search queries.",
    )
    search_results:  Annotated[list[dict], operator.add] = Field(default=None,
        description="Results of the online search"
    ) 
    final_report: Annotated[list, operator.add] = Field(default=None,
        description="The final report"
    )
    is_satisfied : bool = Field(default=False,
        description="true if the report is satisfactory"
    )
    enrichment_steps_taken : int = Field(default=0,
        description="Number of enrichment Steps taken"
    )
    
class UserInput(BaseModel):
    company: str = Field(
        description="The name of the company we are conducting the research on"
    )
    user_focus_topics : list[str] = Field(default=None,
        description="The focus topics for research on the company based only on the user's input"
    )

class Queries(BaseModel):
    queries: list[str] = Field(
        description="List of search queries."
    )

class EnrichmentReport(BaseModel):
    is_satisfied : bool = Field(
        description="true if the report is satisfactory"
    )
    enrichment_queries : list[str] = Field(
        description="If is_satisfied is False, provide 1-3 targeted search queries to find the missing information and to enreach the data"
    )
    reasoning: str = Field(description="Brief explanation of the assessment")

class WebSearchRouter(BaseModel):
    is_search_results_answers_queries: bool = Field(
        description="If the model have enough data based on the search results to answer the queries return True, else return False "
    )
    reasoning: str = Field(description="Brief explanation of the assessment")
