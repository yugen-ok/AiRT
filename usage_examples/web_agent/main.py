"""
Demonstrates basic agent with tf-idf vector search retireval.

Download data from: https://www.kaggle.com/datasets/wcukierski/enron-email-dataset
"""


import json
import random
from pprint import pprint

from airt.Tool import TfIdfVectorSearchTool
from airt.Agent import Agent

from airt.utils.data_utils import *

INPUT_PATH = "data/emails.csv"
JSON_PATH = "output/emails.json"
DB_PATH = "output/emails_vdb.pkl"

emails = []

if os.path.exists(JSON_PATH):
    emails = json.load(open(JSON_PATH))
    print(f"Loaded {len(emails)} emails from {JSON_PATH}")

else:

    with open(INPUT_PATH, encoding="utf8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            emails.append(normalize_row(row))

    with open(JSON_PATH, "w", encoding="utf8") as f:
        json.dump(emails, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(emails)} emails → {JSON_PATH}")

emails = random.sample(emails, 1000)

# This is how we use just the retrieval:
bodies = [email['body'] for email in emails]
# res = run_with_tfidf_vector_search("This is the enron dataset. Find some stuff about top management", bodies)

# ---------------------------------------------------------
# Load documents (you provide this)
# ---------------------------------------------------------

docs = bodies


# ---------------------------------------------------------
# Define retrieval tool (same idea as Tool demo)
# ---------------------------------------------------------

vector_search_tool = TfIdfVectorSearchTool(
    docs=docs,
    save_path=DB_PATH
)


# ---------------------------------------------------------
# Create Agent
# ---------------------------------------------------------

agent = Agent(
    model="gpt-4.1",
    # model="gemini-2.5-flash",
    # model='claude-3-opus-20240229',
    retrieve_tools=[vector_search_tool],
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