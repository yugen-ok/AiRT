
from airt.query_llm import query_llm
from airt.Tool import TfIdfVectorSearchTool, VectorSearchInput

docs = [
    "LangGraph is a framework for building agent workflows.",
    "Vector databases allow semantic search over documents.",
    "TF-IDF is a classical information retrieval technique.",
]

# -------------------------
# Define tool metadata
# -------------------------
tool = TfIdfVectorSearchTool(
    docs=docs
)

# -------------------------
# Ask LLM to decide tool call
# -------------------------
result = query_llm(
    system_prompt="Use the provided tool if it helps answer the user request.",
    user_inputs=["Search my vector DB for langgraph."],
    model="gpt-4.1",
    # model='claude-3-opus-20240229',
    # model='gemini-2.5-flash',
    tools=[tool],
    api_key=anth_api_key
)

tool_name = result.get("tool_name")
tool_args = result.get("arguments")

# -------------------------
# Execute local TF-IDF search
# -------------------------
result = tool.run(
    VectorSearchInput(**tool_args)
)

# Print the first result
print(result.matches[0].content)
