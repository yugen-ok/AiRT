"""

Demonstrates basic agent with SQL-based retrieval.

Download data from: https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis/data


"""


from airt.Tool import SQLDBTool
from airt.Agent import Agent

INPUT_PATH = "data/"
DB_PATH = "output/instacart.sql"

sql_tool = SQLDBTool(directory=INPUT_PATH,
                     save_path=DB_PATH)

print(sql_tool.description)
# ---------------------------------------------------------
# Create Agent
# ---------------------------------------------------------

agent = Agent(
    model="gpt-4.1",
    # model="gemini-2.5-flash",
    tools=[sql_tool],
)

# ---------------------------------------------------------
# Run Agent
# ---------------------------------------------------------

result = agent.run(
    "which departments do we have"
)

print(result)


# Think about how to integrate vector based search with graph based search to identify interesting things about the org.
# Power structure and coord