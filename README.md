# AiRT — A Minimal, Inspectable Runtime for LLM-Driven Internal Tools

## Motivation

Most “LLM agent” frameworks focus on **novelty**: autonomous agents, multi-agent orchestration, planning loops, self-reflection, etc.  
In real projects, teams rarely need that.

What they *actually* need is much simpler, and much harder to get right:

- Deterministic behavior
- Inspectable context
- Schema-stable outputs
- Clear failure modes
- Replaceable components (models, retrieval, tools)
- The ability to debug *why* an answer was produced

In practice, companies end up rebuilding the same missing layer over and over:
a **thin, controllable runtime** that lets LLMs safely sit inside real systems.

Examples:

- Large organizations building internal LLM tools consistently report that **generic chatbots or agent frameworks are insufficient** for production use. Teams end up writing their own orchestration, retrieval, validation, and execution layers to make LLMs usable inside real systems [1][2].

- Instacart’s internal assistant **Ava** began as a hackathon prototype but evolved into a tightly integrated internal productivity system, because simply exposing an LLM interface was not enough for real developer workflows [3].

- J.P. Morgan Payments built **PRBuddy**, a custom internal LLM tool for pull-request workflows, specifically to meet enterprise requirements around determinism, integration with existing tooling, and developer trust — requirements not addressed by generic frameworks [4].

- Industry analyses of enterprise AI adoption show a consistent gap between **LLM experimentation** and **production-ready infrastructure**, with most teams lacking the internal runtime layers needed for reliability, observability, and long-term maintenance [5].


**AiRT exists to be that layer.**

---

## What This Project Is and Isn't

**AiRT is:**
- A minimal runtime for building LLM-powered internal tools
- Focused on *retrieval → reasoning → structured output*
- Explicit, typed, inspectable, and debuggable
- Model-agnostic (OpenAI, Gemini, Claude, etc.)
- Designed for real enterprise workflows, not demos

**AiRT is not:**

- A chatbot framework
- A prompt collection
- An autonomous multi-agent system
- A UI product
- A replacement for business logic

---

## Core Design Principles

### 1. Explicit Structure Over “Agent Magic”
Every step is explicit:
- Retrieval is a tool
- Tool inputs are typed
- Tool outputs are structured
- Final answers are schema-validated

Nothing “just happens”.

---

### 2. Retrieval Is a First-Class Operation
Most enterprise failures are *context failures*, not reasoning failures.

AiRT treats retrieval as:
- A concrete tool
- With a declared input schema
- With a local, testable implementation
- Fully inspectable outside the LLM

---

### 3. Hard Output Contracts
If you ask for structured output, you either get:
- Valid structured output  
**or**
- A hard failure

No silent drift. No best-effort parsing.

---

### 4. Model-Agnostic by Default
The runtime does not care *which* LLM you use.
You can switch models without rewriting your application logic.

---

## High-Level Architecture

```

Task (string)
↓
[ Retrieve Node ]
→ LLM selects a tool
→ Tool executes locally
→ Structured retrieval result
↓
[ Respond Node ]
→ LLM synthesizes an answer
→ Output is schema-validated
↓
[ Eval Node ]
→ (placeholder for scoring / veto / QA)

````

This graph is explicit and inspectable.

---

## Core Components

### `Tool`
A declarative wrapper around an operation the LLM may invoke.

Each tool defines:
- `name`
- `description`
- `input_schema` (Pydantic model)
- `impl` (local runtime implementation)

The LLM only ever sees the *schema + description*.
Execution is fully local.

---

### Retrieval Tools (Examples Included)

- **TF-IDF Vector Search**
  - Local
  - Persistent
  - Rebuildable
  - No external dependencies

- **SQL Database Tool**
  - Auto-builds DB from CSV/XLSX
  - Exposes schema to the LLM
  - Executes parameterized queries safely

Both can be used *without* the LLM for testing.

---

### `Agent`
A minimal execution graph with three nodes:
- `retrieve`
- `respond`
- `eval` (intentionally empty placeholder)

The agent:
- Selects tools via `query_llm`
- Validates tool inputs
- Executes tools locally
- Enforces output schemas strictly

---

## Why This Matters in Real Projects

This design directly addresses common enterprise pain points:

- **Debuggability** → every step is inspectable
- **Reliability** → schemas enforce contracts
- **Safety** → local execution, no hidden actions
- **Maintainability** → no tangled prompt logic
- **Reversibility** → easy to swap models or tools

This is the layer teams usually realize they need *after* shipping a prototype.

---

## Basic Usage

### 1. Prepare Documents (Example: Emails)

```python
docs = ["doc1", "doc2", ...]
````

---

### 2. Create a Retrieval Tool

```python

from airt.Tool import TfIdfVectorSearchTool

vector_search_tool = TfIdfVectorSearchTool(
    docs=docs,
    save_path="output/emails_vdb.pkl"
)
```

---

### 3. Create an Agent

```python

from airt.Agent import Agent

agent = Agent(
  model="gpt-4.1",
  retrieve_tools=[vector_search_tool],
)
```

---

### 4. Run a Task

```python
result = agent.run(
    "Who is asking whom to schedule meetings?"
)

print(result)
```

The result will either:

* Match the declared output schema
* Or raise a clear error

## Output Schemas (Optional but Recommended)

If `inst/output_schema.json` exists:

* The agent **requires** structured output
* Output is validated with JSON Schema
* Invalid output fails fast

This is critical for downstream automation.

---


## Third-Party API Keys


`utils.llm_utils.query_llm()` accepts an `api_key` parameter, and if your environment contains any of the keys:

* `OPENAI_API_KEY`
* `GEMINI_API_KEY`
* `ANTHROPIC_API_KEY`

it will recognize them automatically.

---

## Planned Features:

AiRT intentionally leaves space for:

* Iterative reasoning about retrieved results until satisfied
* Web search
* Caching the output of each step for traceability
* Hierarchical topic-based filtering
* Evaluation / scoring
* Abstention policies
* Context-aware chunking and vectorization
* Image processing
* Human-in-the-loop
* Task-specific retrieval
* Permission-awareness

These are *product decisions*, not framework defaults.

---

## Summary

AiRT is an **intentionally boring, strict, minimal runtime** for using LLMs inside real systems.



## Sources

[1] First Round Review – *The Dynamic Context Problem: How Carta’s Internal AI Agents Save Thousands of Hours*  
https://www.firstround.com/ai/carta

[2] Google People + AI Research – *Building the Plane While Flying It: An LLM Case Study*  
https://medium.com/people-ai-research/building-the-plane-while-flying-it-an-llm-case-study-d5952eec817a

[3] Instacart Tech Blog – *Scaling Productivity with Ava, Instacart’s Internal AI Assistant*  
https://tech.instacart.com/scaling-productivity-with-ava-instacarts-internal-ai-assistant-ed7f02558d84

[4] J.P. Morgan Payments Developer Blog – *How J.P. Morgan Developers Leverage AI*  
https://developer.payments.jpmorgan.com/blog/guides/ai-software-development

[5] ITPro – *AI Tools Are a Game Changer for Enterprise Productivity, but Reliability Issues Persist*  
https://www.itpro.com/business/business-strategy/ai-tools-are-a-game-changer-for-enterprise-productivity-but-reliability-issues-are-causing-major-headaches