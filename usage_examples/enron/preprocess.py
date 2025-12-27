import os
import re
import pandas as pd
import csv
import sys
import quopri

# =========================
# Configuration
# =========================

INPUT_CSV = "data/emails.csv"
OUTPUT_CSV = "output/email_chunks.csv"
DOCS_PATH = "output/email_docs.txt"

MAX_CHARS = 1000
OVERLAP_CHARS = 100

SENTENCE_BOUNDARY_REGEX = re.compile(r'(?<=[\.\?\!])')

# All headers that exist
ALL_HEADERS = [
    "Message-ID", "Date", "From", "To", "Cc", "Bcc", "Subject",
    "Mime-Version", "Content-Type", "Content-Transfer-Encoding",
    "X-From", "X-To", "X-cc", "X-bcc", "X-Folder", "X-Origin", "X-FileName",
    "Attendees", "Re", "Time", "Conference room",
]

# The ones we actually need
CANONICAL_HEADERS = [
    "Message-ID",
    "Date",
    "From",
    "To",
    "Cc",
    "Subject",
    "X-Folder",
]


# =========================
# Helpers (strict)
# =========================

def fix_broken_soft_wraps(text: str) -> str:
    """
    Fix artifacts like 'manage= ment' caused by broken quoted-printable soft wraps.
    Only joins letters split by '= ' or '=\n'.
    """
    return re.sub(r'([A-Za-z])=\s+([A-Za-z])', r'\1\2', text)


def decode_quoted_printable(text: str) -> str:
    """
    Decode quoted-printable encoding in email bodies.
    Safe, lossless, and deterministic.
    """
    return quopri.decodestring(text).decode("utf8", errors="replace")


def debug_codepoints(label: str, s: str, limit: int = 300):
    print(f"\n===== DEBUG: {label} =====")
    for i, ch in enumerate(s[:limit]):
        cp = ord(ch)
        print(
            f"{i:04d}  {repr(ch):>6}  U+{cp:04X}  "
            f"isspace={ch.isspace()}"
        )
    print("===== END DEBUG =====\n")


def collapse_repeated_chars(s: str) -> str:
    result = []

    prev_class = None
    repeat_count = 0

    for ch in s:
        # classify character
        if ch.isdigit():
            char_class = ("digit", ch)
        elif ch == "\n":
            char_class = ("newline", None)
        elif ch.isspace():
            char_class = ("hspace", None)
        else:
            char_class = ("char", ch)

        # repetition logic (class-based)
        if char_class == prev_class:
            repeat_count += 1
        else:
            prev_class = char_class
            repeat_count = 1

        # emit logic
        if char_class[0] == "digit":
            result.append(ch)

        elif char_class[0] == "newline":
            if repeat_count <= 2:
                result.append("\n")

        elif char_class[0] == "hspace":
            if repeat_count == 1:
                result.append(" ")

        else:  # other non-digit chars
            if repeat_count <= 3:
                result.append(ch)

    return "".join(result)


def normalize_body(text: str) -> str:
    # Replace hard line breaks inside paragraphs with spaces
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    return text

def normalize_headers(headers: dict) -> dict:
    return {
        "thread_id": headers.get("Message-ID", ""),
        "date": headers.get("Date", ""),
        "from": headers.get("From", ""),
        "to": headers.get("To", ""),
        "cc": headers.get("Cc", ""),
        "subject": headers.get("Subject", ""),
        "folder": headers.get("X-Folder", ""),
    }


def parse_headers_and_body(raw_message: str):
    """
    Strict RFC-style header parsing.
    Raises ValueError on malformed headers.
    """
    header_block, body = raw_message.split("\n\n", 1)

    headers = {k: "" for k in CANONICAL_HEADERS}

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
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    sentences = []
    for p in paragraphs:
        parts = SENTENCE_BOUNDARY_REGEX.split(p)
        sentences.extend(parts)
    return sentences



def chunk_sentences(sentences):
    chunks = []
    current = ""

    for s in sentences:
        if len(current) + len(s) <= MAX_CHARS:
            if current:
                current = current.rstrip()
                s = s.lstrip()
                current += " " + s
            else:
                current = s
        else:
            chunks.append(current.strip())
            if OVERLAP_CHARS:
                overlap = current[-OVERLAP_CHARS:]
                overlap = overlap.lstrip()
                overlap = overlap.split(" ", 1)[-1]
                overlap = overlap.lstrip()
            # drop partial word
            else:
                overlap = ""

            current = overlap
            if current and not current.endswith(" "):
                current += " "
            current += s

    if current.strip():
        chunks.append(current.strip())

    return chunks



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

    TEXT_COL = "chunk"

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


def build_chunks(load_existing=True):
    """
    End-to-end preprocessing pipeline.

    - If OUTPUT_CSV exists:
        * load it
        * convert to text docs
        * return docs

    - Otherwise:
        * parse raw emails
        * chunk bodies
        * normalize for SQL safety
        * write OUTPUT_CSV
        * convert to text docs
        * return docs
    """

    if os.path.exists(OUTPUT_CSV) and load_existing:
        print(f"[LOAD] Existing chunks detected at {OUTPUT_CSV}")
        return
    # -------------------------
    # Build from scratch
    # -------------------------
    print(f"[START] Building chunks from {INPUT_CSV}")


    rows = []
    dropped = 0

    csv.field_size_limit(sys.maxsize)

    with open(INPUT_CSV, encoding="utf8", newline="") as f:
        reader = csv.DictReader(f)

        for email_idx, row in enumerate(reader):
            if email_idx % 1000 == 0:
                print(f"[PROGRESS] Email {email_idx}")

            raw_msg = row["message"]

            try:
                raw_headers, body = parse_headers_and_body(raw_msg)
                headers = normalize_headers(raw_headers)
            except ValueError:
                dropped += 1
                continue

            # 1. Decode real quoted-printable
            body = decode_quoted_printable(body)

            # 2. Fix broken soft wraps like "manage= ment"
            body = fix_broken_soft_wraps(body)

            # 3. Normalize paragraph structure
            body = normalize_body(body)

            # 4. Collapse pathological repetition
            body = collapse_repeated_chars(body)

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

    df = pd.DataFrame(rows).fillna("")

    # -------------------------
    # FINAL SQL-SAFE NORMALIZATION
    # -------------------------
    INDEX_COLS = {"email_idx", "chunk_idx"}

    for col in df.columns:
        if col not in INDEX_COLS:
            df[col] = df[col].astype(str)
            df[col] = df[col].replace({
                "None": "",
                "nan": "",
                "NaN": ""
            })

    df["email_idx"] = df["email_idx"].astype(int)
    df["chunk_idx"] = df["chunk_idx"].astype(int)

    df.to_csv(OUTPUT_CSV, index=False, encoding="utf8")

    print("[DONE] Chunk CSV created")


def build_docs(load_existing=True):

    if os.path.exists(DOCS_PATH) and load_existing:
        print(f"[LOAD] Loading existing docs from {DOCS_PATH}")

        with open(DOCS_PATH, encoding="utf8") as f:
            docs = f.read().split("\n\n---DOC---\n\n")

        print(f"[DONE] Loaded {len(docs)} documents")
        return docs
    else:
        print(f"[LOAD] Loading chunks from {OUTPUT_CSV} to build docs")

        df = pd.read_csv(OUTPUT_CSV, encoding="utf8", low_memory=False)

        docs = df_chunks_to_text_docs(df)
        print(f"[DONE] Built {len(docs)} documents")

        print(f"[WRITE] Writing docs to {DOCS_PATH}")
        os.makedirs(os.path.dirname(DOCS_PATH), exist_ok=True)
        with open(DOCS_PATH, "w", encoding="utf8") as f:
            f.write("\n\n---DOC---\n\n".join(docs))

        return docs

if __name__ == "__main__":

    # Rebuild
    build_chunks(load_existing=False)
    docs = build_docs(load_existing=False)

    # Sanity check
    import random

    for i in random.sample(len(docs), 20):
        print(docs[i]);print('---------------')