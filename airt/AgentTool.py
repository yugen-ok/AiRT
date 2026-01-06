"""
AgentTool with recursive, readable capability schema.

- Invocation schema is strict and minimal
- Capability schema is embedded structurally
- Fully recursive (AgentTool inside AgentTool)
- Pydantic v2 safe
"""

from typing import Any, Dict
from pydantic import BaseModel, Field, PrivateAttr
import json

from .Tool import Tool
from .Agent import Agent


# =========================================================
# Invocation schema (what the LLM provides)
# =========================================================

class DelegateTask(BaseModel):
    task: str = Field(
        description=(
            "Natural-language task delegated to the agent. "
            "The agent autonomously decides which tools to use."
        )
    )

class AgentInvocation(BaseModel):
    delegate: DelegateTask


# =========================================================
# Capability schema builders (recursive)
# =========================================================

def tool_capability(tool: Tool) -> Dict[str, Any]:
    """
    Produce a readable, structured capability description for a Tool.
    """
    base = {
        "description": tool.description,
        "input_schema": (
            tool.input_schema.model_json_schema()
            if tool.input_schema is not None
            else None
        ),
    }

    # Recursive case: AgentTool
    if isinstance(tool, AgentTool):
        base["agent"] = agent_capability(tool._agent)

    return base


def agent_capability(agent: Agent) -> Dict[str, Any]:
    """
    Produce a recursive capability spec for an Agent.
    """
    tools = {}

    if agent.retrieve_tools:
        for name, tool in agent.retrieve_tools.items():
            tools[name] = tool_capability(tool)

    return {
        "tools": tools,
        "output_schema": agent.output_schema,
        "max_retrieve_steps": agent.max_retrieve_steps,
    }


# =========================================================
# AgentTool schema (invocation + capabilities)
# =========================================================

class AgentToolSchema(BaseModel):
    """
    Full schema exposed to the LLM.

    - invocation: what the model must supply
    - capabilities: what the agent can do (read-only)
    """
    invocation: Dict[str, Any]
    capabilities: Dict[str, Any]


# =========================================================
# AgentTool
# =========================================================

class AgentTool(Tool):
    """
    Tool wrapper around an Agent with recursive schema exposure.
    """

    _agent: Agent = PrivateAttr()

    def __init__(
        self,
        *,
        name: str,
        description: str,
        agent: Agent,
    ):
        # Build full capability tree BEFORE BaseModel init
        schema = AgentToolSchema(
            invocation=AgentInvocation.model_json_schema(),
            capabilities={
                "agent": agent_capability(agent)
            }
        )

        super().__init__(
            name=name,
            description=description,
            input_schema=type(
                f"{name}_Input",
                (BaseModel,),
                {
                    "__annotations__": {
                        "delegate": DelegateTask,
                        "capabilities": Dict[str, Any],
                    },
                    "capabilities": Field(
                        default=schema.capabilities,
                        description=(
                            "READ-ONLY capability description of the agent. "
                            "Use this to decide what task to delegate."
                        ),
                    ),
                },
            ),
            impl=self,
        )

        object.__setattr__(self, "_agent", agent)

    # =====================================================
    # Execution
    # =====================================================

    def run(self, input) -> Any:
        return self._agent.run(input.delegate.task)

