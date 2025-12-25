import csv
import sys
import os
from email.utils import parsedate_to_datetime



# -------------------------------------------------
# Allow very large email fields (Enron-specific)
# -------------------------------------------------
csv.field_size_limit(sys.maxsize)

# -------------------------------------------------
# Step 1: parse raw email into headers + body
# -------------------------------------------------
def parse_enron_message(raw_message: str):
    # Split headers and body at first blank line
    parts = raw_message.split("\n\n", 1)

    header_block = parts[0]
    body = parts[1].strip() if len(parts) > 1 else ""

    headers = {}
    for line in header_block.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            headers[key.strip()] = value.strip()

    return headers, body


# -------------------------------------------------
# Step 2: helpers
# -------------------------------------------------
def split_emails(value):
    if not value:
        return []
    return [x.strip() for x in value.split(",") if x.strip()]

def split_names(value):
    if not value:
        return []
    # remove exchange routing junk
    name = value.split("<")[0].replace("'", "").strip()
    return [name] if name else []

def parse_date(value):
    if not value:
        return None
    try:
        return parsedate_to_datetime(value).isoformat()
    except Exception:
        return value


# -------------------------------------------------
# Step 3: normalize one CSV row into flat dict
# -------------------------------------------------
def normalize_row(row):
    headers, body = parse_enron_message(row["message"])

    folder = headers.get("X-Folder", "")
    folder = os.path.basename(folder) if folder else None

    return {
        "file": row["file"],
        "message_id": headers.get("Message-ID"),
        "date": parse_date(headers.get("Date")),
        "from": headers.get("From"),
        "from_name": headers.get("X-From"),
        "to": split_emails(headers.get("To")),
        "to_names": split_names(headers.get("X-To")),
        "cc": split_emails(headers.get("X-cc")),
        "bcc": split_emails(headers.get("X-bcc")),
        "subject": headers.get("Subject", ""),
        "body": body,
        "folder": folder,
        "mailbox_owner": headers.get("X-Origin"),
        "source_file": headers.get("X-FileName"),
    }

