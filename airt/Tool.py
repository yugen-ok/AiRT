"""
Tool module for AiRT (AI Research Toolkit).

Defines the Tool abstraction for LLM function calling and provides concrete
implementations for common retrieval tasks.

Key concepts:
    - Tool: Base class for defining functions LLMs can call
    - Input/Output schemas: Pydantic models for type safety and validation
    - Local tools: Execute locally (vs. API-based tools)
    - Tool.impl: Runtime implementation attached to Tool object

Provided tools:
    - TfIdfVectorSearchTool: TF-IDF-based vector search
    - SQLDBTool: SQL querying over CSV/Excel data

Example:
     # Define a vector search tool
     docs = ["Document 1", "Document 2"]
     tool = TfIdfVectorSearchTool(docs=docs, save_path="vdb.pkl")
    
     # Use in agent
     agent = Agent(model="gpt-4", tools=[tool], inst_dir="default_inst/")
     result = agent.run("Find information about authentication")
    
     # Or execute directly
     from airt import VectorSearchInput
     output = tool.run(VectorSearchInput(query="auth", top_k=3))
     print(output.matches[0].content)
"""

from typing import Type, List, Dict, Any, Optional
from pydantic import BaseModel, ConfigDict, PrivateAttr, Field
import json

from openai import OpenAI
from .DB import TfIdfVectorDB, SQLDB


# =========================================================
# Base Tool
# =========================================================

class Tool(BaseModel):
    """
    Base class for LLM-callable tools/functions.

    Tools define functions that LLMs can invoke during agent execution.
    Each tool has a name, description, input schema (for validation), and
    optional runtime implementation.

    Attributes:
        name: Function name (shown to LLM)
        description: What the tool does (helps LLM decide when to use it)
        input_schema: Pydantic model defining expected arguments
        impl: Runtime implementation object with .run() method
        save_path: Optional path for persisting tool state

    The Tool class is serializable (for passing to LLM APIs) but the impl
    is excluded from serialization since it's only needed at runtime.

    Example:
         class MyToolInput(BaseModel):
             query: str
        
         tool = Tool(
             name="my_function",
             description="Does something useful",
             input_schema=MyToolInput,
         )
    """
    name: str
    description: str
    input_schema: Type[BaseModel] | None

    # Runtime implementation (not serialized, not validated by Pydantic)
    # Must have .run(input) -> output method
    impl: Optional[Any] = None

    # Allow arbitrary types for input_schema (Pydantic class type)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    save_path: Optional[str] = None

# =========================================================
# Native tools
# =========================================================


class NativeWebSearchTool(Tool):
    def __init__(self):
        super().__init__(
            name="web_search",
            description="Search the live web for up-to-date information",
            input_schema=None,
            impl=None,
        )


# =========================================================
# Vector Search Tool Schemas
# =========================================================

class VectorSearchInput(BaseModel):
    """
    Input schema for vector search operations.

    Attributes:
        query: Search query string
        top_k: Number of most similar documents to retrieve
        window: Include ±window neighboring documents for context
    """
    query: str
    top_k: int = 5
    window: int = 0


class VectorMatch(BaseModel):
    """
    Single match result from vector search.

    Attributes:
        id: Unique identifier for the match
        content: Text content of the matched document/chunk
    """
    id: str
    content: str


class VectorSearchOutput(BaseModel):
    """
    Output schema for vector search operations.

    Attributes:
        matches: List of matched documents sorted by relevance
    """
    matches: List[VectorMatch]


# =========================================================
# TF-IDF Vector Search Tool
# =========================================================

class TfIdfVectorSearchTool(Tool):
    """
    Tool for TF-IDF-based vector search over document collections.

    Wraps TfIdfVectorDB as a Tool for LLM function calling. The LLM can
    invoke this tool by generating VectorSearchInput arguments.

    Attributes:
        name: Tool identifier (default: "vector_search")
        description: Tool description shown to LLM
        docs: List of documents to index
        save_path: Path for caching TF-IDF index
        ngram_min: Minimum n-gram size for TF-IDF
        ngram_max: Maximum n-gram size for TF-IDF
        min_df: Minimum document frequency for terms
        chunk_size: Reserved for future chunking
        overlap: Reserved for future chunking
        _vdb: Private TfIdfVectorDB instance

    Example:
         docs = ["Python is a programming language", "Java is also a language"]
         tool = TfIdfVectorSearchTool(docs=docs, ngram_max=2, save_path="index.pkl")
        
         # LLM generates this input:
         search_input = VectorSearchInput(query="programming", top_k=1)
         output = tool.run(search_input)
         print(output.matches[0].content)
        'Python is a programming language'
    """

    name: str = "vector_search"
    description: str = "Query a local TF-IDF vector database."

    # Tool-specific parameters (excluded from serialization)
    docs: list[str] = Field(exclude=True)
    save_path: str | None = Field(default=None, exclude=True)
    ngram_min: int = Field(default=1, exclude=True)
    ngram_max: int = Field(default=3, exclude=True)
    min_df: int = Field(default=1, exclude=True)
    chunk_size: int = Field(default=1000, exclude=True)
    overlap: int = Field(default=200, exclude=True)

    # Private vector database instance
    _vdb: TfIdfVectorDB | None = PrivateAttr(default=None)

    def __init__(self, docs, **kwargs):
        """
        Initialize TF-IDF vector search tool.

        Args:
            docs: List of text documents to index
            **kwargs: Additional Tool parameters (name, description, etc.)
                      and TfIdfVectorDB parameters (ngram_min, save_path, etc.)
        """
        super().__init__(
            docs=docs,
            **kwargs,
            input_schema=VectorSearchInput,  # LLM outputs this format
            impl=self,  # This object handles execution
        )

    def model_post_init(self, __context):
        """
        Initialize vector database after Pydantic validation.

        Called automatically by Pydantic after __init__. Builds or loads
        the TF-IDF index.
        """
        self._vdb = TfIdfVectorDB(
            self.docs,
            save_path=self.save_path,
            ngram_min=self.ngram_min,
            ngram_max=self.ngram_max,
            min_df=self.min_df,
            chunk_size=self.chunk_size,
            overlap=self.overlap,
        )

    def run(self, input: VectorSearchInput) -> VectorSearchOutput:
        """
        Execute vector search query.

        Args:
            input: VectorSearchInput with query, top_k, and window

        Returns:
            VectorSearchOutput with list of matches

        This method is called by the Agent after the LLM selects this tool
        and generates the input arguments.
        """
        # Query underlying TF-IDF database
        raw_results = self._vdb.query(input.query, input.top_k, input.window)

        # Convert to structured output format
        matches = [
            VectorMatch(id=str(i), content=text)
            for i, text in enumerate(raw_results)
        ]
        return VectorSearchOutput(matches=matches)
# =========================================================
# SQL Query Tool Schemas
# =========================================================

class SQLQueryInput(BaseModel):
    """
    Input schema for SQL query operations.

    Attributes:
        sql: SQL query string (use ? for parameterized queries)
        params: Optional list of parameter values for placeholders
    """
    sql: str
    params: Optional[list[Any]] = None


class SQLRow(BaseModel):
    """
    Single row result from SQL query.

    Attributes:
        values: List of column values in row order
    """
    values: list[Any]


class SQLMatch(BaseModel):
    """
    Result set from SQL query.

    Attributes:
        columns: List of column names
        rows: List of data rows
    """
    columns: list[str]
    rows: List[SQLRow]


class SQLQueryOutput(BaseModel):
    """
    Output schema for SQL query operations.

    Attributes:
        matches: List containing single SQLMatch with query results
    """
    matches: List[SQLMatch]


# =========================================================
# SQL Database Tool
# =========================================================

class SQLDBTool(Tool):
    """
    Tool for SQL querying over CSV/Excel data.

    Wraps SQLDB as a Tool for LLM function calling. Automatically imports
    CSV/Excel files into SQLite and exposes schema to LLM for query generation.

    The tool description is dynamically augmented with the database schema,
    allowing the LLM to generate valid SQL queries against the available tables.

    Attributes:
        name: Tool identifier (default: "sql_query")
        description: Tool description + schema (shown to LLM)
        directory: Directory containing CSV/XLSX files
        db_path: SQLite database path (":memory:" or file path)
        save_path: Optional path for persisting database
        _db: Private SQLDB instance

    Example:
         # Directory contains: sales.csv, products.xlsx
         tool = SQLDBTool(directory="data/", db_path="analytics.db")
        
         # LLM sees schema in description and generates:
         query_input = SQLQueryInput(
             sql="SELECT * FROM sales WHERE revenue > ?",
             params=[1000]
         )
         output = tool.run(query_input)
         print(output.matches[0].columns)
        ['id', 'date', 'revenue']
    """

    name: str = "sql_query"
    description: str = "Query a local SQL database built with the following schema:\n\n"

    # Tool-specific parameters (excluded from serialization)
    directory: str = Field(exclude=True)
    db_path: str = Field(default=":memory:", exclude=True)
    save_path: str | None = Field(default=None, exclude=True)

    # Private SQL database instance
    _db: SQLDB | None = PrivateAttr(default=None)

    def __init__(self, directory: str, **kwargs):
        """
        Initialize SQL database tool.

        Args:
            directory: Path to directory containing CSV/XLSX files
            **kwargs: Additional Tool parameters (name, description, etc.)
                      and SQLDB parameters (db_path, save_path)
        """
        super().__init__(
            directory=directory,
            **kwargs,
            input_schema=SQLQueryInput,  # LLM outputs this format
            impl=self,  # This object handles execution
        )

    def model_post_init(self, __context):
        """
        Initialize SQL database and augment description with schema.

        Called automatically by Pydantic after __init__. Builds or loads
        the database, then appends the schema JSON to the description so
        the LLM knows what tables/columns are available.
        """
        self._db = SQLDB(self.directory, db_path=self.db_path, save_path=self.save_path)

        # Augment description with schema for LLM context
        self.description += json.dumps(self._db.schema())

    def run(self, input: SQLQueryInput) -> SQLQueryOutput:
        """
        Execute SQL query against database.

        Args:
            input: SQLQueryInput with SQL query and optional parameters

        Returns:
            SQLQueryOutput with columns and rows

        This method is called by the Agent after the LLM selects this tool
        and generates the SQL query.
        """
        # Execute query on underlying SQLite database
        cursor = self._db.conn.cursor()
        cursor.execute(input.sql, input.params or ())
        rows = cursor.fetchall()

        # Extract column names from cursor metadata
        columns = [desc[0] for desc in cursor.description] if cursor.description else []

        # Convert to structured output format
        return SQLQueryOutput(
            matches=[SQLMatch(
                columns=columns,
                rows=[SQLRow(values=list(row)) for row in rows]
            )]
        )

