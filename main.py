from fastapi import FastAPI, Request
from pydantic import BaseModel
import os

from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict
from typing import Annotated

# groq_api_key = os.environ.get("GROQ_API_KEY")
llm = ChatGroq(groq_api_key='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', model_name="gemma-7b-it")

class State(TypedDict):
    messages: Annotated[list, add_messages]

graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages": llm.invoke(state['messages'])}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)
graph = graph_builder.compile()

app = FastAPI()

class MessageRequest(BaseModel):
    user_input: str

@app.post("/chat")
async def chat(req: MessageRequest):
    user_msg = req.user_input
    for event in graph.stream({'messages': ("user", user_msg)}):
        for value in event.values():
            return {"response": value["messages"].content}
