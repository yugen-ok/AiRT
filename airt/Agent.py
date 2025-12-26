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

     # Example 1: Agent with retrieval
     docs = ["Document 1 content", "Document 2 content"]
     tool = TfIdfVectorSearchTool(docs=docs, save_path="vdb.pkl")

     # Initialize agent with instructions directory containing:
     # - retrieve.j2: prompt for tool selection
     # - respond.j2: prompt for generating response
     # - output_schema.json: JSONSchema for structured output
     agent = Agent(
         model="gpt-4",
         retrieve_tools=[tool],
         inst_dir="instructions/"
     )

     # Run task
     result = agent.run("What is the main topic of the documents?")

     # Example 2: Agent without retrieval (direct response)
     agent_no_retrieval = Agent(
         model="gpt-4",
         retrieve_tools=None,  # Skip retrieve step
         inst_dir="instructions/"
     )

     # Run task without retrieval - responds directly based on task
     result = agent_no_retrieval.run("Explain the concept of retrieval.")
"""

from typing import List, Any, Optional

from langgraph.graph import StateGraph, END
from pydantic import BaseModel

import json
from pathlib import Path
from jsonschema import validate, ValidationError

from .Tool import Tool
from .utils.utils import jsonschema_to_pydantic_model
from .query_llm import query_llm, render_template




# =========================================================
# Generic Tools
# =========================================================

decide_schema = {
    "type": "object",
    "properties": {
        "action": {
            "type": "string",
            "enum": ["retrieve", "answer"]
        }
    },
    "required": ["action"]
}

DecideModel = jsonschema_to_pydantic_model(
    "ControllerDecision",
    decide_schema
)

decide_tool = Tool(
    name="decide_next",
    description="Decide whether more retrieval is needed or we can answer now",
    input_schema=DecideModel,
)

# =========================================================
# Graph State
# =========================================================

class AgentState(BaseModel):
    """
    State object passed between nodes in the agent execution graph.

    Attributes:
        task: The user's task or query to be processed
        retrieved: Context retrieved from tools (e.g., VectorSearchOutput with matches)
        response: Final structured response from the agent (tool call dict with arguments)
        decision: Latest controller decision ({"action": "retrieve" | "answer"})
        error: Error message if any step fails (None if no errors)
        retrieve_steps: Number of retrieval executions completed (default: 0)
        tool_call_history: List of previous tool calls with their results from retrieve steps
    """
    task: str
    retrieved: Optional[Any]
    response: Optional[Any]
    decision: Optional[dict] = None
    error: Optional[str]
    retrieve_steps: int = 0
    tool_call_history: List[dict] = []



# =========================================================
# Agent
# =========================================================

class Agent:
    """
    LangGraph-based agent for task execution with retrieval and structured outputs.

    The agent follows a graph that adapts based on retrieve_tools:
        - If retrieve_tools is None: respond → eval (no retrieval step)
        - If retrieve_tools is provided: decide ⇄ retrieve → respond → eval
            1. decide: Determines whether to retrieve more context or answer now
            2. retrieve: Uses LLM to select and execute tools for information gathering
            3. respond: Generates structured response based on retrieved context
            4. eval: Placeholder for quality checks and validation

    Attributes:
        model: LLM model identifier (e.g., "gpt-4", "gemini-2.0-flash-exp")
        inst_dir: Directory containing Jinja2 templates and output_schema.json
        retrieve_tools: Dictionary mapping tool names to Tool objects (None if no retrieval)
        local_tool_impls: Dictionary mapping tool names to their .run() implementations (None if no retrieval)
        output_schema: JSONSchema dict for response validation (if output_schema.json exists)
        output_tool: Tool wrapper for structured output (used in respond step)
        graph: Compiled LangGraph StateGraph
        max_retrieve_steps: Maximum number of retrieval iterations allowed
        max_retrieved_chars: Maximum total characters allowed in retrieved context
        verbose: Whether to print debug information
        log_path: Optional path to log file for debug output

    Args:
        model: LLM model identifier
        retrieve_tools: List of Tool objects (must have .impl attribute for local execution).
                       If None, the retrieve step is skipped entirely.
        inst_dir: Path to directory with templates (decide.j2, retrieve.j2, respond.j2)
                  and optional output_schema.json
        max_retrieve_steps: Maximum number of retrieval executions (default: 1)
        max_retrieved_chars: Maximum total characters in retrieved context (default: 8000)
        verbose: Enable debug output to stdout and optional log file (default: False)
        log_path: Path to log file for debug output (requires verbose=True)

    Raises:
        RuntimeError: If task execution fails or output validation fails
    """

    def __init__(
            self,
            model: str,
            retrieve_tools: List[Tool] = None,
            inst_dir: str = "inst/",
            max_retrieve_steps: int = 1,
            max_retrieved_chars: int = 8000,
            verbose: bool = False,
            log_path: Optional[str] = None
    ):

        self.model = model
        self.inst_dir = inst_dir
        self.max_retrieve_steps = max_retrieve_steps
        self.max_retrieved_chars = max_retrieved_chars
        self.verbose = verbose
        self.log_path = log_path
        self._log_file = None

        if self.verbose and self.log_path is not None:
            self._log_file = open(self.log_path, "a", encoding="utf8")

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
            )
        else:
            self.output_schema = self.output_tool = None


        # Tool registry for retrieval step (excludes output_tool)
        # If retrieve_tools is None, skip the retrieve step entirely
        if retrieve_tools is not None:
            self.retrieve_tools = {t.name: t for t in retrieve_tools}

            # Extract local implementations (must be set as .impl on Tool objects)
            self.local_tool_impls = {
                t.name: t.impl for t in retrieve_tools if hasattr(t, "impl")
            }
        else:
            self.retrieve_tools = None
            self.local_tool_impls = None

        self.graph = self._build_graph()

    # =====================================================
    # Graph Construction
    # =====================================================

    def _build_graph(self):
        """
        Constructs the LangGraph execution graph based on retrieval requirements.

        Builds one of two graph structures:
            - No retrieval: respond → eval → END
            - With retrieval: decide ⇄ retrieve → respond → eval → END

        Returns:
            Compiled StateGraph ready for execution
        """
        g = StateGraph(AgentState)

        g.add_node("respond", self._respond)
        g.add_node("eval", self._eval)

        # ---- CASE: no retrieval at all ----
        if self.retrieve_tools is None:
            g.set_entry_point("respond")
            g.add_edge("respond", "eval")
            g.add_edge("eval", END)
            return g.compile()

        # ---- Unified retrieval graph: decide <-> retrieve loop ----
        g.add_node("decide", self._decide)
        g.add_node("retrieve", self._retrieve)

        g.set_entry_point("decide")

        g.add_conditional_edges(
            "decide",
            lambda state: state.decision["action"],
            {
                "retrieve": "retrieve",
                "answer": "respond",
            }
        )

        g.add_edge("retrieve", "decide")
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

        self._debug(
            "RETRIEVE / INPUT STATE",
            task=state.task,
            retrieved=(
                state.retrieved.model_dump()
                if hasattr(state.retrieved, "model_dump")
                else state.retrieved
            )
        )

        # Render prompt from template
        template_context = {
            "task": state.task,
            "tool_call_history": state.tool_call_history
        }

        prompt = render_template(
            f"{self.inst_dir}/retrieve.j2",
            template_context,
        )

        self._debug(
            "RETRIEVE / PROMPT",
            prompt=prompt
        )

        # LLM selects tool and generates arguments
        tool_call = query_llm(
            system_prompt="",
            user_inputs=[prompt],
            model=self.model,
            tools=list(self.retrieve_tools.values()),  # Passes Tool metadata (name, description, schema)
            require_tool=True  # Forces tool call (no free-text responses)
        )

        self._debug(
            "RETRIEVE / TOOL CALL",
            tool_call=tool_call
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

        self._debug(
            "RETRIEVE / TOOL RESULT",
            result=(
                result.model_dump()
                if hasattr(result, "model_dump")
                else result
            )
        )

        # Record this tool call in history
        tool_call_record = {
            "tool_name": name,
            "arguments": args,

        }

        # Accumulate matches from this retrieval with previous matches
        if state.retrieved is None:
            # First retrieval - use result as-is
            accumulated_retrieved = result
        else:
            # Subsequent retrievals - merge matches
            accumulated_retrieved = state.retrieved
            if hasattr(result, "matches") and hasattr(accumulated_retrieved, "matches"):
                # Append new matches to existing ones
                accumulated_retrieved.matches = accumulated_retrieved.matches + result.matches

        update = {
            "retrieved": accumulated_retrieved,
            "retrieve_steps": state.retrieve_steps + 1,
            "tool_call_history": state.tool_call_history + [tool_call_record]
        }

        if self.verbose and "retrieved" not in update:
            print("[WARN] RETRIEVE did not update `retrieved`")

        return update

    def _decide(self, state: AgentState):
        """
        Decide node: Determines whether to retrieve more context or proceed to answer.

        Implements hard limits to prevent excessive retrieval:
            1. Max retrieve steps: Forces "answer" if retrieve_steps >= max_retrieve_steps
            2. Max retrieved chars: Forces "answer" if total chars >= max_retrieved_chars
            3. First retrieval: Forces "retrieve" if no context has been gathered yet
            4. LLM decision: Asks LLM whether more retrieval is needed when within limits

        Args:
            state: Current AgentState with task, retrieved context, and retrieve_steps count

        Returns:
            Dict with "decision" key containing {"action": "retrieve" | "answer"}
        """
        # Hard stop: never exceed max_retrieve_steps retrieval executions
        if state.retrieve_steps >= self.max_retrieve_steps:
            decision = {"action": "answer"}
            self._debug(
                "DECIDE / HARD STOP (STEPS)",
                task=state.task,
                retrieve_steps=state.retrieve_steps,
                max_retrieve_steps=self.max_retrieve_steps,
                decision=decision
            )
            return {"decision": decision}

        # Hard stop: never exceed max_retrieved_chars total character length
        current_chars = 0
        if state.retrieved is not None:
            # Assuming matches have a 'content' or similar text field; 
            # calculating total length of all match dumps for a conservative estimate.
            current_chars = sum(len(str(m)) for m in state.retrieved.matches)

        if current_chars >= self.max_retrieved_chars:
            decision = {"action": "answer"}
            self._debug(
                "DECIDE / HARD STOP (CHARS)",
                task=state.task,
                current_chars=current_chars,
                max_retrieved_chars=self.max_retrieved_chars,
                decision=decision
            )
            return {"decision": decision}

        # If retrieval is still allowed and we have no retrieved context yet, force the first retrieval.
        if state.retrieved is None:
            decision = {"action": "retrieve"}
            self._debug(
                "DECIDE / FIRST RETRIEVE",
                task=state.task,
                retrieve_steps=state.retrieve_steps,
                max_retrieve_steps=self.max_retrieve_steps,
                decision=decision
            )
            return {"decision": decision}

        # Otherwise, we already have context and we still have remaining retrieval budget.
        # Now ask the LLM whether more retrieval is needed or we can answer.
        self._debug(
            "DECIDE / INPUT STATE",
            task=state.task,
            retrieve_steps=state.retrieve_steps,
            max_retrieve_steps=self.max_retrieve_steps,
            matches=(
                [m.model_dump() for m in state.retrieved.matches]
                if state.retrieved is not None
                else []
            )
        )

        prompt = render_template(
            f"{self.inst_dir}/decide.j2",
            {
                "task": state.task,
                "matches": (
                    [m.model_dump() for m in state.retrieved.matches]
                    if state.retrieved is not None
                    else []
                )
            }
        )

        self._debug(
            "DECIDE / PROMPT",
            prompt=prompt
        )

        result = query_llm(
            system_prompt="",
            user_inputs=[prompt],
            model=self.model,
            tools=[decide_tool],
            require_tool=True
        )

        self._debug(
            "DECIDE / OUTPUT",
            result=result
        )

        validate(
            instance=result["arguments"],
            schema=decide_schema
        )

        # Note: retrieve_steps is incremented in _retrieve, not here.
        update = {"decision": result["arguments"]}

        if self.verbose and "decision" not in update:
            print("[WARN] DECIDE did not update `decision`")

        return update

    def _respond(self, state: AgentState):
        """
        Respond node: Generates structured response based on retrieved context (if available).

        Flow:
            1. Renders respond.j2 template with task and retrieved matches (if retrieval occurred)
            2. Calls LLM with output_tool (wraps output schema as tool)
            3. Validates response against JSONSchema (if output_schema.json exists)
            4. Returns validated response

        Args:
            state: Current AgentState with task and optionally retrieved context

        Returns:
            Dict with "response" key containing structured output conforming to schema
            or "error" key if validation fails

        Note:
            Uses JSONSchema validation (not just Pydantic) to ensure strict contract
            compliance across model versions. This prevents silent schema drift.
            If retrieve_tools is None, the retrieved context will be None and respond
            will work without it.
        """

        self._debug(
            "RESPOND / INPUT STATE",
            task=state.task,
            matches=(
                [m.model_dump() for m in state.retrieved.matches]
                if state.retrieved is not None
                else []
            )
        )

        # Build template context based on whether retrieval occurred
        template_context = {"task": state.task}

        if state.retrieved is not None:
            template_context["matches"] = [m.model_dump() for m in state.retrieved.matches]
        else:
            template_context["matches"] = []

        # Render prompt with task and retrieved context (if any)
        prompt = render_template(
            f"{self.inst_dir}/respond.j2",
            template_context,
        )

        self._debug(
            "RESPOND / PROMPT",
            prompt=prompt
        )

        # LLM generates structured response using output_tool
        result = query_llm(
            system_prompt="",
            user_inputs=[prompt],
            model=self.model,
            tools=[self.output_tool] if self.output_tool else None,
        )

        self._debug(
            "RESPOND / RAW OUTPUT",
            result=result
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
                self._debug(
                    "RESPOND / VALIDATED OUTPUT",
                    arguments=result["arguments"]
                )

            except ValidationError as e:
                return {
                    "error": f"Model output does not match output_schema: {e.message}"
                }

        update = {"response": result}

        if self.verbose and "response" not in update:
            print("[WARN] RESPOND did not update `response`")

        return update

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

    def _debug(self, title: str, **payload):
        """
        Outputs debug information to stdout and optional log file.

        Args:
            title: Section title for the debug output
            **payload: Key-value pairs to display (auto-formatted as JSON when possible)

        Note:
            Only outputs when verbose=True. Writes to both stdout and log_path if configured.
        """
        if not self.verbose:
            return

        lines = []
        lines.append("\n" + "=" * 80)
        lines.append(f"[{title}]")
        lines.append("-" * 80)

        for key, value in payload.items():
            lines.append(f"\n{key}:")
            try:
                lines.append(json.dumps(value, indent=2, ensure_ascii=False))

            except Exception:
                lines.append(str(value))
        lines.append("=" * 80)

        text = "\n".join(lines)

        # Print to stdout
        print(text)

        # Write to log file if enabled
        if self._log_file is not None:
            self._log_file.write(text + "\n")
            self._log_file.flush()

    def __del__(self):
        """
        Destructor: Safely closes log file if it was opened.

        Note:
            Uses try-except to prevent errors during interpreter shutdown.
        """
        if getattr(self, "_log_file", None):
            try:
                self._log_file.close()
            except Exception:
                pass

    # =====================================================
    # Public API
    # =====================================================

    def run(self, task: str):
        """
        Execute the agent workflow for a given task.

        Executes the appropriate graph based on configuration:
            - With retrieval: decide ⇄ retrieve → respond → eval
            - Without retrieval: respond → eval

        The graph enforces hard limits (max_retrieve_steps, max_retrieved_chars) and
        validates output against output_schema.json if present.

        Args:
            task: User's task/query as a string

        Returns:
            Structured response dict conforming to output_schema (if defined),
            or raw LLM output if no schema is specified. Format:
                {
                    'tool_name': 'final_answer',
                    'arguments': { ... }  # Conforms to output_schema
                }

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
            "decision": None,
            "retrieve_steps": 0,
            "tool_call_history": [],
        }

        # Invoke graph (synchronous execution)
        result = self.graph.invoke(initial_state)

        # Return response or raise error
        if result.get("response") is not None:
            return result["response"]

        raise RuntimeError(result.get("error", "Unknown failure"))
