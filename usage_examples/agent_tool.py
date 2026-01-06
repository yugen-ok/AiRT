
from airt.Agent import Agent
from airt.Tool import TfIdfVectorSearchTool
from airt.query_llm import query_llm
from airt.AgentTool import AgentTool

from pprint import pprint

docs = [
    "LangGraph is a framework for building agent workflows.",
    "Vector databases allow semantic search over documents.",
    "TF-IDF is a classical information retrieval technique.",
]  * 1000

tfidf_tool = TfIdfVectorSearchTool(docs=docs)

agent = Agent(
    model="gpt-4.1",
    retrieve_tools=[tfidf_tool],
    inst_dir="instructions/"
)

tfidf_agent_tool = AgentTool(
    name="tfidf_agent",
    description="An agent that can answer questions about a database.",
    agent=agent
)

pprint(tfidf_agent_tool.input_schema.model_json_schema())

# The tool's name and description alongside the input_schema will be passed into the llm within query_llm
# when it is called with agent_tool as a tool

response = query_llm(
    user_inputs=["What do the documents say about LangGraph?"],
    model="gpt-4",
    tools=[tfidf_agent_tool],
    require_tool=True
)

print(response)