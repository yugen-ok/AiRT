
import pandas as pd
import inspect
import sys
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent

from agent_code_explorer.source_tracer import print_source
from source_tracer import capture_runtime_sources

# Create a simple sample dataframe
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'Age': [25, 30, 35, 28, 42],
    'City': ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'],
    'Salary': [70000, 80000, 90000, 75000, 95000]
}
df = pd.DataFrame(data)


print("Sample DataFrame:")
print(df)
print("\n" + "="*50 + "\n")

# Create the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)



# Create the pandas dataframe agent
# IMPORTANT: Set allow_dangerous_code=True (required for code execution)
agent = create_pandas_dataframe_agent(
    llm,
    df,
    verbose=True,
    allow_dangerous_code=True  # Required to execute code
)

#print("Question 1: How many rows are in the dataframe?")
#result = agent.invoke("How many rows are in the dataframe?")
#print(f"Answer: {result['output']}\n")

sources = capture_runtime_sources(agent.invoke, input="How many rows are in the dataframe?")

ag_srcs = [src for src in sources if \
           src['class']==sources[0]['class'] or \
           'RunnableSequence' in src['class'] or \
            'PythonAstREPLTool' in src['class'] or \
           'plan' in src['name']
           ]

with open("agent_sources.txt", "w") as f:
    for src in ag_srcs:
        print_source(src, out=f)

# Create trace:

# Write all key information to a local file
output_file = "agent_execution_trace.txt"

with open(output_file, 'w', encoding='utf-8') as f:
    f.write("=" * 80 + "\n")
    f.write("KEY INVOKE METHODS IN EXECUTION ORDER\n")
    f.write("=" * 80 + "\n")

    for src in sources:
        # Check if this is one of our key invoke methods
        key_invokes = [
            'AgentExecutor.invoke',
            'PromptTemplate.invoke',
            'ChatOpenAI.invoke',
            'ReActSingleInputOutputParser.invoke',
            'RunnableSequence.invoke',
            'RunnableBinding.invoke'
        ]

        if src['name'] and any(key in src['name'] for key in key_invokes):
            f.write(f"\n{'=' * 80}\n")
            f.write(f"Class: {src['class']}\n")
            f.write(f"Method: {src['name']}\n")
            f.write(f"File: {src['file']}\n")
            f.write(f"{'=' * 80}\n")
            f.write(src['source'][:1500] if src['source'] and len(src['source']) > 1500 else src[
                                                                                                 'source'] or 'Source not available')
            f.write("\n\n")

    # Also check for any openai-related methods
    f.write("\n" + "=" * 80 + "\n")
    f.write("OPENAI/LLM RELATED METHODS\n")
    f.write("=" * 80 + "\n")

    for src in sources:
        if src['class'] and (
                'openai' in src['class'].lower() or 'chat' in src['class'].lower() or 'llm' in src['class'].lower()):
            f.write(f"\nClass: {src['class']}\n")
            f.write(f"Method: {src['name']}\n")
            f.write(f"File: {src['file']}\n")
            f.write("-" * 40 + "\n")

    # Check for _generate, _call, generate methods which might be the actual LLM call
    f.write("\n" + "=" * 80 + "\n")
    f.write("POTENTIAL LLM EXECUTION METHODS\n")
    f.write("=" * 80 + "\n")

    for src in sources:
        if src['name'] and any(
                x in src['name'].lower() for x in ['_generate', 'generate', '_call', 'create', 'completion']):
            if src['class'] and 'openai' in src['class'].lower():
                f.write(f"\nClass: {src['class']}\n")
                f.write(f"Method: {src['name']}\n")
                f.write(f"File: {src['file']}\n")
                f.write(src['source'][:800] if src['source'] and len(src['source']) > 800 else src[
                                                                                                   'source'] or 'Source not available')
                f.write("\n" + "-" * 80 + "\n")

print(f"Output written to: {output_file}")

# Method 1: Check the steps in the runnable sequence
print(agent.agent.runnable.steps)

# Method 2: Look at the middle step (usually the prompt)
print(agent.agent.runnable.middle)

# Method 3: Check if there's a prompt attribute
if hasattr(agent.agent.runnable, 'prompt'):
    print(agent.agent.runnable.prompt.template)

# Method 4: Iterate through steps to find PromptTemplate
for i, step in enumerate(agent.agent.runnable.steps):
    print(f"\nStep {i}: {type(step)}")
    if hasattr(step, 'template'):
        print(f"Template:\n{step.template}")
    if hasattr(step, 'prompt'):
        print(f"Prompt:\n{step.prompt.template}")

# Method 5: Check the bound runnable if it exists
if hasattr(agent.agent.runnable, 'bound'):
    print(agent.agent.runnable.bound)

