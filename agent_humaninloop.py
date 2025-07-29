from dotenv import load_dotenv
load_dotenv()
from langgraph.graph import StateGraph ,END
from typing import TypedDict,Annotated
import operator
import google.generativeai as genai
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage,AIMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
import getpass
import os
from uuid import uuid4

# import asyncio
# from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver

# genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
if not os.environ.get("TAVILY_API_KEY"):
    os.environ["TAVILY_API_KEY"] = getpass.getpass("Tavily API key:\n")

tool=TavilySearch(max_results=4)
print(type(tool))
print(tool.name)
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

# For persistence in the agent, we can use SqliteSaver
# conn = sqlite3.connect(":memory:")
conn = sqlite3.connect(":memory:", check_same_thread=False)
memory = SqliteSaver(conn)
# print(type(memory))
# print(type(memory.conn))
def reduce_messages(left: list[AnyMessage], right: list[AnyMessage]) -> list[AnyMessage]:
    # assign ids to messages that don't have them
    for message in right:
        if not message.id:
            message.id = str(uuid4())
    # merge the new messages with the existing messages
    merged = left.copy()
    for message in right:
        for i, existing in enumerate(merged):
            # replace any existing messages with the same id
            if existing.id == message.id:
                merged[i] = message
                break
        else:
            # append any new messages to the end
            merged.append(message)
    return merged

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], reduce_messages]


class Agent:
    def __init__(self,model,tools,checkpointer,system=""):
        self.system=system
        graph=StateGraph(AgentState)
        graph.add_node("llm",self.call_genai)
        graph.add_node("action",self.take_action)
        graph.add_conditional_edges("llm",self.exists_action,{True:"action",False:END})
        graph.add_edge("action","llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile(
            checkpointer=checkpointer,
            interrupt_before=["action"]
        )
        self.tools={t.name:t for t in tools}
        self.model=model.bind_tools(tools)
        # self.chat=ChatGoogleGenerativeAI(model=self.model,system_instruction=self.system_instruction)

    def exists_action(self,state:AgentState):
        
        result=state["messages"][-1]
        print(f"Result: {result}")
        return len(result.tool_calls)>0

    def call_genai(self,state:AgentState):  
        messages=state["messages"]
        # print(f"Message: {messages}")
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
            print(f"Message: {messages}")
        message = self.model.invoke(messages)
        print(f"Messageafter Invoke: {message}")
        return {'messages': [message]}
    
    def take_action(self,state:AgentState):
        tool_calls=state["messages"][-1].tool_calls
        # print(f"Tool calls: {tool_calls}")
        results=[]
        for t in tool_calls:
            print(f"Calling: {t}")
            result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}
prompt = """You are a smart research assistant.Use tools like Tavily search or other tools to get real-time answers. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""

model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")  
abot = Agent(model, [tool], checkpointer=memory,system=prompt)

# messages = [HumanMessage(content="What is the weather in sf?")]
# thread={"configurable":{"thread_id":"1"}}
# # result = abot.graph.stream({"messages": messages})
# # print(result["messages"][-1].content)
# for event in abot.graph.stream({"messages": messages}, thread):
#     for v in event.values():
#         print(v)

# abot.graph.get_state(thread)
# abot.graph.get_state(thread).next
messages = [HumanMessage("Whats the weather in LA?")]
thread = {"configurable": {"thread_id": "2"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)
while abot.graph.get_state(thread).next:
    print("\n", abot.graph.get_state(thread),"\n")
    _input = input("proceed?")
    if _input != "y":
        print("aborting")
        break
    for event in abot.graph.stream(None, thread):
        for v in event.values():
            print(v)

# messages = [HumanMessage(content="What is the weather in la?")]
# thread={"configurable":{"thread_id":"1"}}
# # result = abot.graph.stream({"messages": messages})
# # print(result["messages"][-1].content)
# for event in abot.graph.stream({"messages": messages}, thread):
#     for v in event.values():
#         print(v['messages'])

# messages = [HumanMessage(content="Which one is hotter")]
# thread={"configurable":{"thread_id":"1"}}
# # result = abot.graph.stream({"messages": messages})
# # print(result["messages"][-1].content)
# for event in abot.graph.stream({"messages": messages}, thread):
#     for v in event.values():
#         print(v['messages'])

# messages = [HumanMessage(content="Which one is hotter")]
# thread={"configurable":{"thread_id":"2"}}
# # result = abot.graph.stream({"messages": messages})
# # print(result["messages"][-1].content)
# for event in abot.graph.stream({"messages": messages}, thread):
#     for v in event.values():
#         print(v['messages'])


