# from dotenv import load_dotenv
# from pydantic import BaseModel
# from langchain_openai import ChatOpenAI
# from langchain_anthropic import ChatAnthropic
# # from langchain_community.llms import Ollama
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.output_parsers import PydanticOutputParser
# from langchain.agents import create_tool_calling_agent
# from langchain.agents import AgentExecutor
# from langchain_ollama import OllamaLLM

# load_dotenv()
# class ResearchResponse(BaseModel):
#     topic:str
#     summary:str
#     sources:list[str]
#     tools_used:list[str]    

# llm=OllamaLLM(model="llama3.2")
# print(hasattr(llm, "bind_tools"))  # should return True
# # llm=ChatOpenAI(model="gpt-4o")
# # llm2=ChatAnthropic(model="claude-3-5-sonnet-20241022")
# parser=PydanticOutputParser(pydantic_object=ResearchResponse)

# prompt=ChatPromptTemplate.from_messages(
#     [
#         ("system",
#          """
#         You are a research assistant that will help generate a research paper.
#         Answer the user query  and use necessary tools.
#         Wrap the output in this format and provide no other text\n{format_instructions}

#          """,
        
#         ),
#         ("placeholder","{chat_history}"),
#         ("human","{query}"),
#         ("placeholder","{agent_scratchpad}"),
#     ]
# ).partial(format_instructions=parser.get_format_instructions())

# # agent=create_tool_calling_agent(
# #     llm=llm,
# #     prompt=prompt,
# #     tools=[]
    
# # )
# # agent_executor=AgentExecutor(agent=agent,verbose=True)
# # raw_response=agent_executor.invoke({"query":"what is the capital of france?"})
# formatted_prompt = prompt.format(query="what is the capital of france?")
# raw_response = llm.invoke(formatted_prompt)
# try:

#     parsed = parser.parse(raw_response)
# except Exception as e:
#     parsed = f"Error parsing response: {e}"
# print(parsed)
# # print(raw_response)
# # RESPONSE=llm.invoke("what is the capital of France?")
# # print(RESPONSE)
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain_ollama import OllamaLLM
from typing import List, Optional
from tools import search_tool,wiki_tool



load_dotenv()

# class ResearchResponse(BaseModel):
#     topic: str
#     summary: str
#     sources: list[str]
#     tools_used: list[str]
tools= [search_tool,wiki_tool]
class ResearchResponse(BaseModel):
    topic: str
    summary: str
    sources: Optional[List[str]] = Field(default_factory=list)
    tools_used:Optional[List[str]] = Field(default_factory=tools)


parser = PydanticOutputParser(pydantic_object=ResearchResponse)
instructions = parser.get_format_instructions()

# Build a single‐string prompt that:
#  1) shows the schema, 
#  2) gives an example JSON,
#  3) asks for a new instance,
#  4) and tells the model to output only that JSON.
query=input("what do you want to research?")
# query = "What is the capital of France?"
# Use the search tool to fetch additional information
tool_result = search_tool.func(query)

prompt = f"""
You are a research assistant.  
You must answer the user’s query and produce _only_ valid JSON matching this schema (no extra text):

{instructions}

For example:
{{
  "topic": "Capital of France",
  "summary": "France’s capital city is Paris, located on the Seine river.",
  "sources": ["Wikipedia"],
  "tools_used": ["GeoLookup, DuckDuckGo Search"]
}}

Now, for this query:
User Query: "{query}"
DuckDuckGo Tool Result: "{tool_result}"
Tools Used: [{tools}]
Sources should include the website from where you got the information.
Return just the JSON object.
""".strip()

llm = OllamaLLM(model="llama3.2")
raw = llm.invoke(prompt)

# Extract text if Ollama returned a dict
print("==RAW MODEL OUTPUT==")
print(raw)
text = raw if isinstance(raw, str) else raw.get("completion", raw.get("text", ""))
print("==MODEL OUTPUT==")
print(text)

# Parse into Pydantic model
result = parser.parse(text)
print("\n==PARSED OUTPUT==")
# print(result.json(indent=2))
print(result.model_dump_json())
