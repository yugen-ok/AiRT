
import re
import os
import json
from PyPDF2 import PdfReader

def read_pdf(path):

    assert path.lower().endswith(".pdf")
    print(f"processing {path}")
    reader = PdfReader(path)

    pages_text = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages_text.append(page_text)

    full_text = "\n".join(pages_text)

    return full_text

def chunk_text_sliding(text, chunk_size=1000, overlap=200):
    text = re.sub(r"\s+", " ", text)
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

def chunkify_dir(data_directory_path, chunk_size=1000, overlap=200):
    merged = ""

    for filename in os.listdir(data_directory_path):
        filepath = os.path.join(data_directory_path, filename)

        if os.path.isfile(filepath):
            with open(filepath, 'r', encoding='utf8') as f:
                merged += f.read() + "\n"

    print("Chunking text...")
    chunks = chunk_text_sliding(merged, chunk_size, overlap)
    print("Chunks created:", len(chunks))

    print("Building TF-IDF index...")

    return chunks