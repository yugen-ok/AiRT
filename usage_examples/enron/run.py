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
from airt.utils.doc_utils import chunk_text_sliding


INPUT_PATH = "data/emails.csv"
JSON_PATH = "output/emails.json"
DB_PATH = "output/emails_vdb.pkl"

# ---------------------------------------------------------
# Chunk emails into normal-sized documents with metadata
# ---------------------------------------------------------

def chunk_emails(emails, chunk_size=2000):
    """
    Create documents from emails where each document is a chunk with full email metadata.

    Returns:
        List of dicts, each containing email metadata and a body chunk
    """
    documents = []

    for email in emails:
        body_chunks = chunk_text_sliding(email.get('body', ''), chunk_size, overlap=20)

        for chunk_idx, chunk in enumerate(body_chunks):
            doc = {
                'message_id': email.get('message_id'),
                'date': email.get('date'),
                'from': email.get('from'),
                'from_name': email.get('from_name'),
                'to': email.get('to'),
                'to_names': email.get('to_names'),
                'cc': email.get('cc'),
                'bcc': email.get('bcc'),
                'subject': email.get('subject'),
                'folder': email.get('folder'),
                'mailbox_owner': email.get('mailbox_owner'),
                'source_file': email.get('source_file'),
                'body_chunk': chunk,
                'chunk_index': chunk_idx,
                'total_chunks': len(body_chunks)
            }
            documents.append(doc)

    return documents

# ---------------------------------------------------------
# Load and chunk data
# ---------------------------------------------------------


emails = []

if os.path.exists(JSON_PATH):
    emails = json.load(open(JSON_PATH))
    print(f"Loaded {len(emails)} emails from {JSON_PATH}")

else:

    with open(INPUT_PATH, encoding="utf8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            emails.append(normalize_row(row))

    os.makedirs(os.path.dirname(JSON_PATH), exist_ok=True)
    with open(JSON_PATH, "w", encoding="utf8") as f:
        json.dump(emails, f, indent=2, ensure_ascii=False)

    print(f"Converted {len(emails)} emails → {JSON_PATH}")

# Subsample (optional)
emails = random.sample(emails, 1000)


email_chunks = chunk_emails(emails, chunk_size=500)
print(f"Created {len(email_chunks)} document chunks from {len(emails)} emails")

# ---------------------------------------------------------
# Build tool
# ---------------------------------------------------------

# Create searchable text from each document (for TF-IDF indexing)
docs = [
    f"Email Chunk #{doc['chunk_index']}\n"
    f"From: {doc['from_name'] or doc['from']}\n"
    f"To: {', '.join(doc['to_names']) if doc['to_names'] else ', '.join(doc['to'] or [])}\n"
    f"Date: {doc['date']}\n"
    f"Subject: {doc['subject']}\n\n"
    f"{doc['body_chunk']}"
    for doc in email_chunks
]


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
    max_retrieve_steps=5,
    max_retrieved_chars = 10000,
    verbose=True,
    log_path="agent_debug.log"
)

# ---------------------------------------------------------
# Run Agent
# ---------------------------------------------------------

result = agent.run(
    "Who is asking whom to schedule meetings? Collect at least 5 examples"
)

pprint(result)


# Think about how to integrate vector based search with graph based search to identify interesting things about the org.
# Power structure and coord