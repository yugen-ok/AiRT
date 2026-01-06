"""
Demonstrates basic agent with tf-idf vector search retireval.

Download data from: https://www.kaggle.com/datasets/wcukierski/enron-email-dataset
"""


from pprint import pprint

from preprocess import build_chunks, build_docs

from airt.Tool import TfIdfVectorSearchTool, SQLDBTool
from airt.Agent import Agent

OUTPUT_DIR = "output/"

SQL_DB_PATH = f"{OUTPUT_DIR}/emails.sql"
VECTOR_DB_PATH = f"{OUTPUT_DIR}/emails_vdb.pkl"

build_chunks()
docs = build_docs()

sql_tool = SQLDBTool(
    directory=OUTPUT_DIR,
    save_path=SQL_DB_PATH,
)

tfidf_tool = TfIdfVectorSearchTool(
    docs=docs,
    save_path=VECTOR_DB_PATH
)


# ---------------------------------------------------------
# Create Agent
# ---------------------------------------------------------

agent = Agent(
    model="gpt-4.1",
    # model="gemini-2.5-flash",
    # model='claude-3-opus-20240229',
    retrieve_tools=[tfidf_tool],
    max_retrieve_steps=3,
    verbose=True,
    log_path="output/agent_debug.log",
)

# ---------------------------------------------------------
# Run Agent
# ---------------------------------------------------------
query = """
Find at least 10 examples of someone asking for a discussion
 to move offline and are not within a marketing email.

Marketing emails are completely irrelevant for this task.

Consider multiple phrases for coverage.

Query for phrases like:

call me
let’s discuss offline
not in email
verbal discussion
pick up the phone

Limit tok_k to a small number to avoid explosion. 

Cite the source as [<email_idx>.<chunk_idx>], using the corresponding email_idx and chunk_idx from the header fields.

"""

result = agent.run(
    query
)

pprint(result)


# Think about how to integrate vector based search with graph based search to identify interesting things about the org.
# Power structure and coord