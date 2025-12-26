"""
This is the basic use case of Tool+query_llm without the Agent class,
and without actual function calls. This demonstrates how to use Tool
for output schema definition and enforcement only.

This script defines a tool, which gives the LLM a structured output schema,
and then queries the LLM with a user input, which is then parsed into the schema.

"""


# -------------------------
# Minimal output schema (title, date, location only)
# -------------------------
from pydantic import BaseModel

from airt.Tool import Tool
from airt.query_llm import query_llm


# -------------------------
# Define input / output schemas
# -------------------------

class CreateEventInput(BaseModel):
    title: str
    date: str  # YYYY-MM-DD
    location: str


class CreateEventOutput(BaseModel):
    title: str
    date: str
    location: str


# -------------------------
# Define the Tool
# -------------------------

create_event_tool = Tool(
    name="create_event",
    description="Create a simple calendar event",
    input_schema=CreateEventInput,
)

# -------------------------
# Query OpenAI via unified interface
# -------------------------

result = query_llm(
    system_prompt="Extract event information and return it as structured data.",
    user_inputs=[
        "Create an event called AI Meetup in Berlin on 2025-03-10"
    ],
    model="gpt-4.1",
    # model="gemini-2.5-flash",
    # model='claude-3-opus-20240229',
    temperature=0,
    tools=[create_event_tool]
)

print(result)