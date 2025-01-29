from typing import List
from typing_extensions import TypedDict
from pydantic import BaseModel, Field
from IPython.display import Image, display
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from state import AgentState
from typing_extensions import TypedDict
from langchain_core.messages import AnyMessage
from typing import Annotated, List
import operator




llm = ChatOpenAI(model="gpt-4o", temperature=0) 

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]

def process_query(state: AgentState):
    user_input = input("Enter your input: ")  # Get input outside the function
    response = llm.invoke(user_input)  # Get LLM's response
    state['messages'] = [response]  # Store the message in the state
    return state

def decide_mood(state: AgentState):
    user_input = state.get('messages')
    print(user_input)
    if user_input[0] == 'end' :
        return END
    return "process_query"

builder = StateGraph(AgentState)

builder.add_node("process_query", process_query)

builder.add_edge(START,"process_query")
builder.add_conditional_edges("process_query",decide_mood)
builder.add_edge("process_query",END)


graph = builder.compile()

# while True:
result = graph.invoke(input("Hi! Ask something: "))  # Get user input and start the graph
print(result)
    # if result == END:
    #     print("Conversation ended.")
    #     break

