"""
Demonstrates basic agent with tf-idf vector search retireval.

Download data from: https://www.kaggle.com/datasets/wcukierski/enron-email-dataset
"""


import json
import random
import pandas as pd
from pprint import pprint

from preprocess import build_chunks, build_docs

from airt.Tool import TfIdfVectorSearchTool, SQLDBTool
from airt.Agent import Agent

SQL_DB_PATH = "output/emails.sql"
VECTOR_DB_PATH = "output/emails_vdb.pkl"


build_chunks()
docs = build_docs()

sql_tool = SQLDBTool(
    directory="output/",
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
    max_retrieve_steps=10,
    max_retrieved_chars = 10000,
    verbose=True,
    log_path="agent_debug.log"
)

# ---------------------------------------------------------
# Run Agent
# ---------------------------------------------------------
query = """
Find some examples of decisions that happen “offline”
Phrases like:

“Let’s discuss verbally”

“Better not put this in email”

“Call me”

Limit tok_k to a small number.

Cite the source as [<email_idx>.<chunk_idx>], using the corresponding email_idx and chunk_idx from the header fields.

"""

result = agent.run(
    query
)

pprint(result)


# Think about how to integrate vector based search with graph based search to identify interesting things about the org.
# Power structure and coord