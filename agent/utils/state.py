from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from typing import Annotated, List
import operator


class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


