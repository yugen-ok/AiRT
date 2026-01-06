"""
Database module for AiRT (AI Research Toolkit).

Provides persistent, queryable data structures for context retrieval:
    - TfIdfVectorDB: Sparse vector search using TF-IDF + cosine similarity
    - SQLDB: Structured data from CSV/XLSX files with SQL querying
    - BertFaissVectorDB: Dense embeddings with FAISS for semantic search

All DB classes support:
    - Lazy initialization (build on first use or load from cache)
    - Persistence (save/load to disk)
    - Rebuild flag for forcing fresh construction

Example:
     # TF-IDF vector search
     docs = ["Machine learning basics", "Neural networks explained"]
     vdb = TfIdfVectorDB(docs, save_path="index.pkl")
     results = vdb.query("What is backpropagation?", top_k=3)

     # SQL database from CSVs
     sqldb = SQLDB(directory="data/", db_path="analytics.db")
     rows = sqldb.query("SELECT * FROM sales WHERE revenue > ?", [1000])
"""

import os
import pickle
import numpy as np
import re

import sqlite3
from pathlib import Path
import shutil

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

class DB:
    """
    Abstract base class for persistent, queryable databases.

    Implements lazy initialization pattern:
        - On init, attempts to load from save_path if it exists
        - Otherwise, builds from scratch via _build_from_scratch()
        - Saves result to disk if save_path is provided

    Attributes:
        save_path: File path for persistence (None = no persistence)
        _rebuild: If True, forces rebuild even if cache exists

    Subclasses must implement:
        - _build_from_scratch(): Construction logic
        - query(): Query interface
        - save(): Serialization logic (extends base behavior)
        - load(): Deserialization logic (extends base behavior)
    """
    def __init__(self, save_path: str = None, rebuild: bool = False):
        """
        Initialize database with optional persistence.

        Args:
            save_path: Path for saving/loading DB (None = ephemeral)
            rebuild: Force rebuild even if cached version exists
        """
        self.save_path = save_path
        self._rebuild = rebuild
        self._build()

    # -------------------------
    # Lifecycle Management
    # -------------------------
    def _build(self):
        """
        Ensures DB is ready for queries.

        Flow:
            1. If save_path exists and rebuild=False, load from cache
            2. Otherwise, build from scratch via _build_from_scratch()
            3. If save_path is set, persist the built DB

        This pattern avoids expensive rebuilds during development.
        """
        if self.save_path and os.path.exists(self.save_path) and not self._rebuild:
            self.load()
        else:
            self._build_from_scratch()
            if self.save_path:
                self.save()

    # -------------------------
    # Abstract Methods (Subclass Interface)
    # -------------------------
    def _build_from_scratch(self):
        """
        Construct DB from raw data sources.

        Subclasses MUST implement this to define how the DB is built
        (e.g., fit TF-IDF vectorizer, load CSVs into SQLite, encode embeddings).
        """
        raise NotImplementedError

    def query(self, prompt):
        """
        Query interface for retrieving information.

        Args:
            prompt: Query string or parameters

        Returns:
            Query results (format varies by subclass)

        Subclasses MUST implement this.
        """
        raise NotImplementedError

    def save(self):
        """
        Persist DB to disk at save_path.

        Base implementation creates parent directories. Subclasses should
        call super().save() then add their serialization logic.
        """
        print("Saving DB to:", self.save_path)
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)

    def load(self):
        """
        Load DB from disk at save_path.

        Base implementation logs the load. Subclasses should call super().load()
        then add their deserialization logic.
        """
        print("Loading DB from:", self.save_path)


class TfIdfVectorDB(DB):
    """
    Sparse vector database using TF-IDF for text retrieval.

    Uses sklearn's TfidfVectorizer with all configuration supplied externally.
    This class applies TF-IDF exactly as configured by the calling Tool and
    does not define policy-level defaults.
    Suitable for keyword-based retrieval where exact term matches matter.

    Advantages:
        - Fast (no GPU required)
        - Interpretable (can inspect which terms matched)
        - Good for technical/domain-specific vocabulary

    Disadvantages:
        - No semantic understanding (won't match synonyms)
        - Requires careful tuning of min_df and n-gram ranges

    Attributes:
        docs: List of text strings to index
        vectorizer: TfidfVectorizer instance
        matrix: Sparse TF-IDF matrix (docs × vocab)

    Example:
         docs = ["Machine learning is a subset of AI",
                 "Deep learning uses neural networks"]
         vdb = TfIdfVectorDB(docs, ngram_min=1, ngram_max=2, save_path="index.pkl")
         results = vdb.query("What is deep learning?", top_k=1, window=0)
         print(results[0])
        'Deep learning uses neural networks'
    """

    def __init__(
            self,
            docs,
            *,
            stop_words,
            ngram_min,
            ngram_max,
            min_df,
            max_df,
            sublinear_tf,
            norm,
            token_pattern,
            save_path=None,
            rebuild=False,
    ):

        """
        Initialize TF-IDF vector database.

        Args:
            docs: List of text strings to index
            ngram_min: Minimum n-gram size (1 = unigrams)
            ngram_max: Maximum n-gram size (2 = bigrams, 3 = trigrams, etc.)
            min_df: Minimum document frequency (filters rare terms)
            save_path: Path for caching fitted vectorizer and matrix
            rebuild: Force rebuild even if cache exists
        """

        # Configure TF-IDF with n-grams for better phrase matching
        self.vectorizer = TfidfVectorizer(
            analyzer="word",
            stop_words=stop_words,
            ngram_range=(ngram_min, ngram_max),
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=sublinear_tf,
            norm=norm,
            token_pattern=token_pattern,
        )

        self.docs = docs
        self.matrix = None  # Populated in _build_from_scratch()

        super().__init__(save_path, rebuild)

    def _build_from_scratch(self):
        """
        Fit TF-IDF vectorizer on documents.

        Creates sparse matrix where rows = documents, columns = vocabulary.
        Each cell contains TF-IDF weight for that term in that document.
        """
        print("Building TF-IDF matrix...")
        self.matrix = self.vectorizer.fit_transform(self.docs)
        print("Complete.")

    def get_top_inds(self, text, top_k):
        """
        Retrieve indices of top-k most similar documents.

        Args:
            text: Query string
            top_k: Number of documents to retrieve

        Returns:
            NumPy array of document indices sorted by cosine similarity (descending)
        """
        # Transform query into same vector space as documents
        q_vec = self.vectorizer.transform([text])

        # Compute cosine similarity: query × all documents
        scores = cosine_similarity(q_vec, self.matrix).flatten()

        # Return top-k indices (highest scores first)
        top_indices = scores.argsort()[::-1][:top_k]
        return top_indices

    def query(self, prompt, top_k=5, window=0):
        """
        Query database for relevant documents with optional context window.

        Args:
            prompt: Query string
            top_k: Number of documents to retrieve
            window: Expand results to include ±window surrounding documents
                    (useful for maintaining context across split documents)

        Returns:
            List of document strings sorted by original document order

        Example:
             vdb.query("neural networks", top_k=2, window=1)
            # Returns docs at indices [4, 5, 6] if doc 5 was top match
            # (includes neighbors at 4 and 6 for context)
        """
        inds = self.get_top_inds(prompt, top_k)

        # Expand to include neighboring documents for context
        all_inds = set()
        for i in inds:
            start = max(0, i - window)
            end = min(len(self.docs), i + window + 1)
            for j in range(start, end):
                all_inds.add(j)

        # Sort indices to preserve sequential order (important for narrative coherence)
        sorted_inds = sorted(all_inds)

        # Return documents in order
        return [self.docs[i] for i in sorted_inds]

    def save(self):
        """
        Persist vectorizer, documents, and TF-IDF matrix to disk.

        Saves all three components as a pickle to enable fast loading
        without refitting.
        """
        super().save()
        with open(self.save_path, "wb") as f:
            pickle.dump((self.docs, self.vectorizer, self.matrix), f)

    def load(self):
        """
        Load previously saved TF-IDF database from disk.

        Restores documents, fitted vectorizer, and sparse matrix.
        """
        super().load()
        with open(self.save_path, "rb") as f:
            self.docs, self.vectorizer, self.matrix = pickle.load(f)


class SQLDB(DB):
    """
    SQL database built from CSV/XLSX files for structured data retrieval.

    Automatically imports all CSV and Excel files from a directory into SQLite
    tables. Each file becomes a table (Excel sheets become separate tables).

    Use cases:
        - Structured data analysis (sales, logs, surveys)
        - Join operations across multiple datasets
        - Aggregations and filtering via SQL

    Attributes:
        directory: Path to folder containing CSV/XLSX files
        db_path: SQLite database path (":memory:" for in-memory DB)
        conn: sqlite3.Connection object

    Example:
         # Directory contains: sales.csv, products.xlsx (with 2 sheets)
         db = SQLDB(directory="data/", db_path="analytics.db")
         db.tables()  # ['sales', 'products__sheet1', 'products__sheet2']
         rows = db.query("SELECT * FROM sales WHERE revenue > ?", [1000])
    """
    def __init__(self, directory, db_path=":memory:",
                 save_path = None, rebuild = False):
        """
        Initialize SQL database from directory of CSV/XLSX files.

        Args:
            directory: Path to folder containing data files
            db_path: SQLite database path (":memory:" = ephemeral in-memory DB,
                     or file path like "data.db" for persistence)
            save_path: Optional separate path for saving DB copy
            rebuild: Force rebuild even if save_path cache exists
        """
        self.directory = Path(directory)
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path)
        super().__init__(save_path, rebuild)

    # ======================================================
    # Public API
    # ======================================================

    def query(self, sql, params=None):
        """
        Execute SQL query with optional parameters.

        Args:
            sql: SQL query string (use ? for placeholders)
            params: Tuple/list of parameter values for placeholders

        Returns:
            List of tuples (one per row)

        Example:
             db.query("SELECT name, price FROM products WHERE price > ?", [50])
            [('Widget', 75.0), ('Gadget', 120.0)]
        """
        cur = self.conn.cursor()
        cur.execute(sql, params or ())
        return cur.fetchall()

    def tables(self):
        """
        List all table names in database.

        Returns:
            List of table name strings
        """
        cur = self.conn.cursor()
        cur.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        )
        return [row[0] for row in cur.fetchall()]

    def schema(self) -> dict:
        """
        Get database schema as JSON-serializable dict.

        Useful for passing schema to LLMs for SQL generation.

        Returns:
            Dict with structure:
            {
                "tables": {
                    "table_name": {
                        "columns": {"col_name": "col_type"}
                    }
                }
            }

        Example:
             db.schema()
            {
                'tables': {
                    'sales': {
                        'columns': {
                            'id': 'INTEGER',
                            'date': 'TEXT',
                            'revenue': 'REAL'
                        }
                    }
                }
            }
        """
        cursor = self.conn.cursor()

        cursor.execute("""
                       SELECT name
                       FROM sqlite_master
                       WHERE type = 'table'
                         AND name NOT LIKE 'sqlite_%'
                       ORDER BY name
                       """)
        tables = [row[0] for row in cursor.fetchall()]

        schema = {"tables": {}}

        for table in tables:
            cursor.execute(f"PRAGMA table_info({table})")
            columns = cursor.fetchall()

            schema["tables"][table] = {
                "columns": {
                    col_name: col_type
                    for _, col_name, col_type, _, _, _ in columns
                }
            }

        return schema

    def _build_from_scratch(self):
        """
        Scan directory and import all CSV/XLSX files into SQLite.

        Each CSV becomes a table; each Excel sheet becomes a separate table
        with naming format: filename__sheetname.
        """
        for file in self.directory.iterdir():
            if file.suffix.lower() == ".csv":
                self._load_csv(file)
            elif file.suffix.lower() in {".xlsx", ".xls"}:
                self._load_xlsx(file)


    def save(self,):
        """
        Persist database to disk at save_path.

        For in-memory databases, uses SQLite's backup API to write to disk.
        For file-based databases, copies the file to save_path.
        """
        super().save()
        path = Path(self.save_path)

        if self.db_path == ":memory:":
            # Backup in-memory DB to disk
            disk_conn = sqlite3.connect(path)
            self.conn.backup(disk_conn)
            disk_conn.close()
        else:
            # Copy existing DB file if different from save_path
            if Path(self.db_path) != path:
                shutil.copyfile(self.db_path, path)

    def load(self):
        """
        Load database from save_path and replace current connection.

        Closes existing connection and reopens with the loaded database.
        """
        super().load()
        self.conn.close()
        self.db_path = str(self.save_path)
        self.conn = sqlite3.connect(self.db_path)

    # ======================================================
    # File Loaders (Internal)
    # ======================================================

    def _load_csv(self, path):
        """
        Load CSV file into SQLite table.

        Args:
            path: Path to CSV file

        Table name is derived from filename (sanitized to valid SQL identifier).
        """
        import pandas as pd

        table = self._sanitize_name(path.stem)
        df = pd.read_csv(path, dtype=str, low_memory=False).fillna("")
        df.to_sql(table, self.conn, if_exists="replace", index=False)

    def _load_xlsx(self, path):
        """
        Load Excel file into SQLite tables (one per sheet).

        Args:
            path: Path to Excel file

        Each sheet becomes a table named: filename__sheetname (sanitized).
        """
        import pandas as pd

        xls = pd.ExcelFile(path)
        for sheet in xls.sheet_names:
            # Use double underscore to separate filename from sheet name
            table = self._sanitize_name(f"{path.stem}__{sheet}")
            df = xls.parse(sheet)
            df.to_sql(table, self.conn, if_exists="replace", index=False)

    # ======================================================
    # Utilities
    # ======================================================

    def _sanitize_name(self, name):
        """
        Convert filename/sheetname to valid SQL identifier.

        Args:
            name: Original name string

        Returns:
            Sanitized name (lowercase, alphanumeric + underscores only)

        Example:
            _sanitize_name("Sales Data 2023!.csv")
            'sales_data_2023_csv'
        """
        name = name.lower()
        # Replace non-alphanumeric with underscores
        name = re.sub(r"[^a-z0-9_]+", "_", name)
        # Collapse multiple underscores
        name = re.sub(r"_+", "_", name)
        return name.strip("_")


class BertFaissVectorDB(DB):
    """
    Dense semantic vector database using transformer embeddings and FAISS.

    Uses sentence-transformers for neural text encoding and FAISS for efficient
    approximate nearest neighbor search. Superior to TF-IDF for semantic matching.

    Advantages:
        - Semantic understanding (matches synonyms, paraphrases)
        - Language-agnostic (multilingual models available)
        - Handles long queries well

    Disadvantages:
        - Slower than TF-IDF (requires GPU for speed)
        - Less interpretable (black-box embeddings)
        - Larger storage footprint

    Dependencies (lazy-loaded):
        - sentence-transformers: For BERT-style embeddings
        - faiss-cpu or faiss-gpu: For vector search

    Attributes:
        data_directory_path: Directory containing text files to index
        model_name: HuggingFace model ID for embeddings
        normalize: Normalize embeddings for cosine similarity
        docs: List of strings
        embeddings: NumPy array of dense vectors
        index: FAISS index for similarity search

    Example:
         db = BertFaissVectorDB(
             data_directory_path="docs/",
             model_name="all-MiniLM-L6-v2",
             save_path="embeddings.pkl"
         )
         result = db.query("How do I configure authentication?", top_k=3)
    """

    def __init__(
        self,
        data_directory_path,
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        normalize=True,
        save_path=None,
        rebuild=False
    ):
        """
        Initialize BERT+FAISS vector database.

        Args:
            data_directory_path: Directory with text files to encode
            model_name: HuggingFace sentence-transformer model
            normalize: L2-normalize embeddings for cosine similarity
            save_path: Path for caching embeddings and index
            rebuild: Force rebuild even if cache exists
        """
        self.data_directory_path = data_directory_path
        self.model_name = model_name
        self.normalize = normalize

        # Lazy-loaded to avoid importing heavy dependencies unless needed
        self.model = None
        self.faiss = None
        self.index = None

        self.docs = []
        self.embeddings = None
        self.dim = None
        super().__init__(save_path, rebuild)

    # ======================================================
    # Internal Helpers
    # ======================================================

    def _lazy_imports(self):
        """
        Import heavy dependencies only when needed.

        Loads sentence-transformers and FAISS on first use to avoid
        startup overhead when this class isn't instantiated.

        Requires:

        pip install faiss==1.5.3 sentence_transformers==5.2.0


        """
        if self.model is None:
            from sentence_transformers import SentenceTransformer
            import faiss

            self.model = SentenceTransformer(self.model_name)
            self.faiss = faiss

    def _load(self):
        """
        Read all text files from directory into self.docs.
        """
        for filename in os.listdir(self.data_directory_path):
            filepath = os.path.join(self.data_directory_path, filename)
            if os.path.isfile(filepath):
                with open(filepath, "r", encoding="utf8") as f:
                    self.docs.append(f.read())

    def _build_from_scratch(self):
        """
        Build vector database: load files, chunk, encode, and index.

        Steps:
            1. Load sentence-transformer model
            2. Read and chunk all text files
            3. Encode chunks into dense vectors
            4. Normalize vectors (for cosine similarity)
            5. Build FAISS index for fast search
        """
        self._lazy_imports()
        self._load()

        # Encode all docs using transformer model
        embeddings = self.model.encode(
            self.docs,
            convert_to_numpy=True,
            show_progress_bar=True
        )

        # L2-normalize for cosine similarity via inner product
        if self.normalize:
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            embeddings = embeddings / np.clip(norms, 1e-12, None)

        self.embeddings = embeddings.astype("float32")
        self.dim = self.embeddings.shape[1]

        # FAISS flat index (exact search, cosine via inner product)
        self.index = self.faiss.IndexFlatIP(self.dim)
        self.index.add(self.embeddings)

    # ======================================================
    # Public API
    # ======================================================

    def query(self, prompt, top_k=5, window=5):
        """
        Semantic search for relevant docs.

        Args:
            prompt: Query string
            top_k: Number of most similar docs to retrieve
            window: Include ±window neighboring docs for context

        Returns:
            Single merged string of all retrieved docs (sorted by position)

        Example:
             result = db.query("authentication workflow", top_k=3, window=1)
            # Returns text from 3 matched docs plus their neighbors
        """
        self._lazy_imports()

        # Encode query
        q = self.model.encode(
            [prompt],
            convert_to_numpy=True
        )

        # Normalize to match training-time behavior
        if self.normalize:
            q = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)

        q = q.astype("float32")

        # Search FAISS index
        scores, inds = self.index.search(q, top_k)
        inds = inds[0]

        # Expand to include neighboring docs for context
        all_inds = set()
        for i in inds:
            start = max(0, i - window)
            end = min(len(self.docs), i + window + 1)
            for j in range(start, end):
                all_inds.add(j)

        # Merge docs in sequential order (preserves narrative flow)
        merged_text = " ".join(self.docs[i] for i in sorted(all_inds))
        return merged_text

    def save(self):
        """
        Persist embeddings and FAISS index to disk.

        Creates two files:
            - save_path: Pickle with texts, embeddings, and metadata
            - save_path.faiss: FAISS index file
        """
        super().save()
        self._lazy_imports()

        # Save metadata and embeddings as pickle
        with open(self.save_path, "wb") as f:
            pickle.dump(
                {
                    "texts": self.docs,
                    "embeddings": self.embeddings,
                    "dim": self.dim,
                    "model_name": self.model_name,
                    "normalize": self.normalize,
                },
                f
            )

        # Save FAISS index separately (uses custom binary format)
        self.faiss.write_index(self.index, self.save_path + ".faiss")

    def load(self):
        """
        Load previously saved embeddings and FAISS index.

        Restores texts, embeddings, and index from disk to enable
        instant querying without re-encoding.
        """
        self._lazy_imports()

        # Load metadata and embeddings
        with open(self.save_path, "rb") as f:
            data = pickle.load(f)

        self.docs = data["texts"]
        self.embeddings = data["embeddings"]
        self.dim = data["dim"]
        self.model_name = data["model_name"]
        self.normalize = data["normalize"]

        # Load FAISS index
        self.index = self.faiss.read_index(self.save_path + ".faiss")


