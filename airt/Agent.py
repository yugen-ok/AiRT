"""
Agent module for AiRT (AI Research Toolkit).

This module implements a LangGraph-based agent that follows a retrieve→respond→eval
workflow for task execution with structured outputs. The agent uses tools for
information retrieval and enforces schema validation on outputs.

Key concepts:
    - Context Assembly: Deterministically assembles the exact context needed for a task
    - Task-Scoped RAG: Uses per-task retrieval configurations via tools
    - Schema-Stable Outputs: Hard validation ensures output contracts across model versions
    - Traceability: LangGraph provides step-by-step execution traces

Example:

     from airt import Agent, TfIdfVectorSearchTool
    
     # Create a vector search tool
     docs = ["Document 1 content", "Document 2 content"]
     tool = TfIdfVectorSearchTool(docs=docs, save_path="vdb.pkl")
    
     # Initialize agent with instructions directory containing:
     #   - retrieve.j2: prompt for tool selection
     #   - respond.j2: prompt for generating response
     #   - output_schema.json: JSONSchema for structured output
     agent = Agent(
         model="gpt-4",
         tools=[tool],
         inst_dir="instructions/"
     )
    
     # Run task
     result = agent.run("What is the main topic of the documents?")
"""

from typing import List, Any, Optional

from langgraph.graph import StateGraph, END
from pydantic import BaseModel

import json
from pathlib import Path
from jsonschema import validate, ValidationError

from .Tool import Tool
from .utils.utils import jsonschema_to_pydantic_model
from .utils.llm_utils import query_llm, render_template


# =========================================================
# Graph State
# =========================================================

class AgentState(BaseModel):
    """
    State object passed between nodes in the agent execution graph.

    Attributes:
        task: The user's task or query to be processed
        retrieved: Context retrieved from tools (e.g., VectorSearchOutput)
        response: Final structured response from the agent
        error: Error message if any step fails
    """
    task: str
    retrieved: Optional[Any]
    response: Optional[Any]
    error: Optional[str]


# =========================================================
# Agent
# =========================================================

class Agent:
    """
    LangGraph-based agent for task execution with retrieval and structured outputs.

    The agent follows a three-node graph:
        1. retrieve: Uses LLM to select and execute tools for information gathering
        2. respond: Generates structured response based on retrieved context
        3. eval: Placeholder for quality checks and validation

    Attributes:
        model: LLM model identifier (e.g., "gpt-4", "gemini-2.0-flash-exp")
        inst_dir: Directory containing Jinja2 templates and output_schema.json
        retrieve_tools: Dictionary mapping tool names to Tool objects
        local_tool_impls: Dictionary mapping tool names to their .run() implementations
        output_schema: JSONSchema dict for response validation (if output_schema.json exists)
        output_tool: Tool wrapper for structured output (used in respond step)
        graph: Compiled LangGraph StateGraph

    Args:
        model: LLM model identifier
        retrieve_tools: List of Tool objects (must have .impl attribute for local execution)
        inst_dir: Path to directory with templates (retrieve.j2, respond.j2)
                  and optional output_schema.json

    Raises:
        RuntimeError: If task execution fails or output validation fails
    """
    def __init__(
        self,
        model: str,
        retrieve_tools: List[Tool] = None,
        inst_dir: str = "inst/",
    ):
        self.model = model
        self.inst_dir = inst_dir

        # Load output schema if present for hard validation
        schema_path = Path(inst_dir) / "output_schema.json"
        if schema_path.exists():
            with open(schema_path, encoding="utf8") as f:
                output_schema = json.load(f)

            self.output_schema = output_schema

            # Convert JSONSchema to Pydantic for LLM tool calling
            OutputModel = jsonschema_to_pydantic_model("FinalAnswer", self.output_schema)

            # Wrap schema as a Tool for uniform interface in respond step
            # Note: input_schema here represents the output format (naming is counterintuitive
            # but required for tool calling API compatibility)
            self.output_tool = Tool(
                name="final_answer",
                description="Produce the final structured response",
                input_schema=OutputModel,  # This is the actual output schema
                kind="native",
            )
        else:
            self.output_schema = self.output_tool = None


        # Tool registry for retrieval step (excludes output_tool)
        self.retrieve_tools = {t.name: t for t in retrieve_tools}

        # Extract local implementations (must be set as .impl on Tool objects)
        self.local_tool_impls = {
            t.name: t.impl for t in retrieve_tools if hasattr(t, "impl")
        }

        self.graph = self._build_graph()

    # =====================================================
    # Graph Construction
    # =====================================================

    def _build_graph(self):
        """
        Constructs the LangGraph execution graph.

        Creates a linear three-node pipeline:
            retrieve → respond → eval → END

        Returns:
            Compiled StateGraph ready for invocation
        """
        g = StateGraph(AgentState)


        g.add_node("retrieve", self._retrieve)
        g.add_node("respond", self._respond)
        g.add_node("eval", self._eval)

        g.set_entry_point("retrieve")
        g.add_edge("retrieve", "respond")
        g.add_edge("respond", "eval")
        g.add_edge("eval", END)

        return g.compile()

    # =====================================================
    # Graph Nodes
    # =====================================================

    def _retrieve(self, state: AgentState):
        """
        Retrieve node: Uses LLM to select and execute a tool for context gathering.

        Flow:
            1. Renders retrieve.j2 template with task
            2. Calls LLM with available tools (require_tool=True forces tool use)
            3. Validates tool selection and arguments
            4. Executes local tool implementation
            5. Returns retrieved context in state

        Args:
            state: Current AgentState with task

        Returns:
            Dict with "retrieved" key containing tool output (e.g., VectorSearchOutput)
            or "error" key if tool selection/execution fails

        Note:
            The LLM acts as a tool selector, not a traditional agent. This ensures
            deterministic retrieval based on the task description.
        """
        # Render prompt from template
        prompt = render_template(
            f"{self.inst_dir}/retrieve.j2",
            {"task": state.task},
        )

        # LLM selects tool and generates arguments
        tool_call = query_llm(
            system_prompt="",
            user_inputs=[prompt],
            model=self.model,
            tools=list(self.retrieve_tools.values()),  # Passes Tool metadata (name, description, schema)
            require_tool=True  # Forces tool call (no free-text responses)
        )

        # Validate LLM output format
        if not isinstance(tool_call, dict) or "tool_name" not in tool_call or "arguments" not in tool_call:
            return {"error": f"Invalid tool call output: {tool_call}"}

        name = tool_call["tool_name"]
        args = tool_call["arguments"]

        # Verify tool has local implementation
        tool_impl = self.local_tool_impls.get(name)
        if tool_impl is None:
            return {"error": f"No local impl for tool {name}"}

        # Validate arguments against tool's input schema
        tool_def = self.retrieve_tools.get(name)
        if tool_def is None:
            return {"error": f"Unknown tool selected: {name}"}

        # Parse and validate arguments (e.g., VectorSearchInput)
        input_obj = tool_def.input_schema.model_validate(args)

        # Execute tool (e.g., TfIdfVectorDB.query())
        result = tool_impl.run(input_obj)

        return {"retrieved": result}

    def _respond(self, state: AgentState):
        """
        Respond node: Generates structured response based on retrieved context.

        Flow:
            1. Renders respond.j2 template with task and retrieved matches
            2. Calls LLM with output_tool (wraps output schema as tool)
            3. Validates response against JSONSchema (if output_schema.json exists)
            4. Returns validated response

        Args:
            state: Current AgentState with task and retrieved context

        Returns:
            Dict with "response" key containing structured output conforming to schema
            or "error" key if validation fails

        Note:
            Uses JSONSchema validation (not just Pydantic) to ensure strict contract
            compliance across model versions. This prevents silent schema drift.
        """
        if state.retrieved is None:
            return {"error": "Respond called without retrieved context."}

        # Render prompt with task and retrieved context
        prompt = render_template(
            f"{self.inst_dir}/respond.j2",
            {
                "task": state.task,
                "matches": [m.model_dump() for m in state.retrieved.matches],
            },
        )

        # LLM generates structured response using output_tool
        result = query_llm(
            system_prompt="",
            user_inputs=[prompt],
            model=self.model,
            tools=[self.output_tool] if self.output_tool else None,
        )

        # HARD schema validation (enforces output contract)
        if self.output_tool is not None:
            if not isinstance(result, dict):
                return {
                    "error": f"Structured output required but model returned {type(result)}"
                }

            # Validate against JSONSchema (stricter than Pydantic type hints)
            try:
                validate(instance=result["arguments"], schema=self.output_schema)
            except ValidationError as e:
                return {
                    "error": f"Model output does not match output_schema: {e.message}"
                }

        return {"response": result}

    def _eval(self, state: AgentState):
        """
        Eval node: Placeholder for quality checks and response validation.

        Future implementations may include:
            - Completeness checks (did we answer all parts of the task?)
            - Correctness checks (run gold-set validation)
            - Schema diff detection (compare with previous runs)
            - Confidence scoring
            - Automatic retry logic if validation fails

        Currently a no-op that passes state through unchanged.

        Args:
            state: Current AgentState with response

        Returns:
            Empty dict (no state modifications)
        """
        return {}

    # =====================================================
    # Public API
    # =====================================================

    def run(self, task: str):
        """
        Execute the agent workflow for a given task.

        Runs the retrieve→respond→eval pipeline and returns the final structured
        output or raises an error if any step fails.

        Args:
            task: User's task/query as a string

        Returns:
            Structured response dict conforming to output_schema (if defined),
            or raw LLM output if no schema is specified

        Raises:
            RuntimeError: If any graph node returns an error or validation fails

        Example:
             agent.run("What are the key findings about climate change?")
            {
                'tool_name': 'final_answer',
                'arguments': {
                    'findings': ['Finding 1', 'Finding 2'],
                    'confidence': 'high'
                }
            }
        """
        initial_state = {
            "task": task,
            "retrieved": None,
            "response": None,
            "error": None,
        }

        # Invoke graph (synchronous execution)
        result = self.graph.invoke(initial_state)

        # Return response or raise error
        if result.get("response") is not None:
            return result["response"]

        raise RuntimeError(result.get("error", "Unknown failure"))
