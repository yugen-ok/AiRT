import os
import csv
import re
import pandas as pd

# =========================
# Configuration
# =========================

INPUT_CSV = "data/emails.csv"
OUTPUT_CSV = "output/email_chunks.csv"

MAX_CHARS = 500
OVERLAP_CHARS = 50

SENTENCE_BOUNDARY_REGEX = re.compile(r'(?<=[\.\?\!\n])')

# Header keys we allow (measured universe)
HEADER_KEYS = [
    "Message-ID", "Date", "From", "To", "Cc", "Bcc", "Subject",
    "Mime-Version", "Content-Type", "Content-Transfer-Encoding",
    "X-From", "X-To", "X-cc", "X-bcc", "X-Folder", "X-Origin", "X-FileName",
    "Attendees", "Re", "Time", "Conference room",
]

# =========================
# Helpers (strict)
# =========================

def parse_headers_and_body(raw_message: str):
    """
    Strict RFC-style header parsing.
    Raises ValueError on malformed headers.
    """
    header_block, body = raw_message.split("\n\n", 1)

    headers = {k: "" for k in HEADER_KEYS}

    current_key = None

    for line in header_block.splitlines():
        if line.startswith((" ", "\t")):
            if current_key is None:
                raise ValueError("Header continuation without a key")
            headers[current_key] += " " + line.strip()
            continue

        if ":" not in line:
            raise ValueError(f"Malformed header line: {line}")

        key, value = line.split(":", 1)
        key = key.strip()

        if key not in headers:
            # ignore unknown headers deterministically
            current_key = None
            continue

        headers[key] = value.strip()
        current_key = key

    body = body.strip()
    if not body:
        raise ValueError("Empty body")

    return headers, body


def sentence_tokenize(text: str):
    sentences = SENTENCE_BOUNDARY_REGEX.split(text)
    return [s.strip() for s in sentences if s.strip()]


def chunk_sentences(sentences):
    chunks = []
    current = ""

    for s in sentences:
        if len(current) + len(s) <= MAX_CHARS:
            current += s
        else:
            chunks.append(current.strip())
            overlap = current[-OVERLAP_CHARS:] if OVERLAP_CHARS else ""
            current = overlap + s

    if current.strip():
        chunks.append(current.strip())

    return chunks


# =========================
# Main pipeline
# =========================

def build_or_load_chunks():
    if os.path.exists(OUTPUT_CSV):
        print(f"[LOAD] Loading existing chunk CSV: {OUTPUT_CSV}")
        df = pd.read_csv(OUTPUT_CSV, encoding="utf8")
        print(f"[DONE] Loaded {len(df)} chunks into DataFrame")
        return df

    print(f"[START] Building chunk CSV from {INPUT_CSV}")

    rows = []
    dropped = 0

    with open(INPUT_CSV, encoding="utf8", newline="") as f:
        reader = csv.DictReader(f)

        for email_idx, row in enumerate(reader):
            if email_idx % 1000 == 0:
                print(f"[PROGRESS] Email {email_idx}")

            raw_msg = row["message"]

            try:
                headers, body = parse_headers_and_body(raw_msg)
            except ValueError:
                dropped += 1
                continue

            sentences = sentence_tokenize(body)
            chunks = chunk_sentences(sentences)

            for chunk_idx, chunk in enumerate(chunks):
                rows.append({
                    "email_idx": email_idx,
                    "chunk_idx": chunk_idx,
                    "chunk": chunk,
                    "file": row["file"],
                    **headers,
                })

    print(f"[SUMMARY] Dropped malformed emails: {dropped}")
    print(f"[WRITE] Writing {len(rows)} chunks to {OUTPUT_CSV}")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_CSV, index=False, encoding="utf8")

    print("[DONE] Chunk CSV created and loaded into DataFrame")
    return df

def df_chunks_to_text_docs(df):
    """
    Convert df_chunks into a list of text documents.

    Each document consists of:
    - A metadata header (key: value per line)
    - A blank line
    - The body chunk text

    Returns:
        List[str]
    """

    TEXT_COL = "body_chunk"

    # ---- hard schema checks ----
    assert TEXT_COL in df.columns, (
        f"Expected text column '{TEXT_COL}' not found. "
        f"Available columns: {list(df.columns)}"
    )

    docs = []

    metadata_cols = [c for c in df.columns if c != TEXT_COL]

    for _, row in df.iterrows():
        header_lines = []

        for col in metadata_cols:
            val = row[col]
            if pd.isna(val):
                val = ""
            header_lines.append(f"{col}: {val}")

        header = "\n".join(header_lines)
        body = row[TEXT_COL]

        docs.append(f"{header}\n\n{body}")

    return docs


# =========================
# Execute
# =========================

if __name__ == "__main__":
    df_chunks = build_or_load_chunks()
    docs = df_chunks_to_text_docs(df_chunks)