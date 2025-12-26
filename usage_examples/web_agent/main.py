
import json
import random
from pprint import pprint

from airt.Tool import TfIdfVectorSearchTool
from airt.Agent import Agent

from airt.utils.data_utils import *

agent = Agent(
    model="gpt-4.1",
    # model="gemini-2.5-flash",
    # model='claude-3-opus-20240229',
)

# ---------------------------------------------------------
# Run Agent
# ---------------------------------------------------------

result = agent.run(
    "Who is asking whom to schedule meetings?"
)

pprint(result)


# Think about how to integrate vector based search with graph based search to identify interesting things about the org.
# Power structure and coord