
from airt.query_llm import query_llm
from airt.Tool import NativeWebSearchTool


# -------------------------
# Define tool
# -------------------------

tool = NativeWebSearchTool()

# -------------------------
# Ask LLM to decide tool call
# -------------------------
result = query_llm(
    system_prompt="Latest news from france with dates",
    # model="gpt-4.1",
    model='gemini-2.5-flash',
    # model='claude-3-opus-20240229',
    tools=[tool],
)

print(result)
