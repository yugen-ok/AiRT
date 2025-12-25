"""
This is an example of how to use tool calls with OpenAI.
This does not use anything from AiRT, and the idea is to
demonstrate the functionality that it abstracts over.

airt.Tool is basically an abstraction layer over this
functionality, which allows you to fold the tool's schema
and actual function into a single object
"""


from openai import OpenAI
import json

client = OpenAI()

# -------------------------------------------------
# 1. Define a function tool (Responses API format)
# -------------------------------------------------
tools = [
    {
        "type": "function",
        "name": "add",
        "description": "Add two integers",
        "parameters": {
            "type": "object",
            "properties": {
                "a": {"type": "integer"},
                "b": {"type": "integer"},
            },
            "required": ["a", "b"],
        },
    }
]

# -------------------------------------------------
# 2. Ask the model (it will call the function)
# -------------------------------------------------
response = client.responses.create(
    model="gpt-4.1",
    input="What is 7 plus 5?",
    tools=tools,
)

# -------------------------------------------------
# 3. Extract the function call
# -------------------------------------------------
function_call = next(
    item for item in response.output if item.type == "function_call"
)

args = json.loads(function_call.arguments)
result = args["a"] + args["b"]

print(result)
