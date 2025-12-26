"""
LLM utilities for AiRT (AI Research Toolkit).

Provides a unified, provider-agnostic interface for querying LLMs with:
    - Automatic routing based on model name (gpt-*, gemini-*, claude-*)
    - Disk-based caching for reproducibility and cost savings
    - Async batching with concurrency control
    - Tool calling support across all providers
    - Consistent return semantics (scalar for single, list for multiple)

Key features:
    - query_llm(): Main entry point for all LLM calls
    - render_template(): Jinja2 template rendering for prompts
    - Automatic retry and error handling
    - Provider-specific tool format conversion

Supported providers:
    - OpenAI (GPT-4, GPT-4o, o1, etc.)
    - Google Gemini (gemini-1.5-pro, gemini-2.0-flash, etc.)
    - Anthropic Claude (claude-3-5-sonnet, etc.)

Example:
     # Simple text completion
     result = query_llm(
         system_prompt="You are a helpful assistant.",
         user_inputs=["What is 2+2?"],
         model="gpt-4",
         temperature=0.0
     )
     print(result)  # "4"

     # Tool calling
     from airt import Tool, VectorSearchInput
     tools = [vector_search_tool]
     result = query_llm(
         user_inputs=["Find docs about authentication"],
         model="gemini-2.0-flash-exp",
         tools=tools,
         require_tool=True
     )
     print(result)  # {"tool_name": "vector_search", "arguments": {}}
"""

# -*- coding: utf-8 -*-
import asyncio
import hashlib
import json
import os
from typing import List, Optional, Union, Any

import diskcache
import openai
from google import genai
from google.genai import types
import anthropic
from jinja2 import Template

def render_template(path, args):
    """
    Render a Jinja2 template file with provided arguments.

    Args:
        path: Path to .j2 or .jinja2 template file
        args: Dict of variables to pass to template

def render_template(path, args):

    Returns:
        Rendered string

    Example:
         render_template("prompts/respond.j2", {"task": "Summarize", "matches": []})
        'Based on the following context:\\n'
    """
    # Load template from file
    with open(path, encoding='utf8') as f:
        template = Template(f.read())

    # Render with provided arguments
    result = template.render(args)
    return result


def _hash_request(obj: dict) -> str:
    """
    Generate deterministic cache key from request parameters.

    Uses SHA256 hash of sorted JSON to ensure same requests hit cache.

    Args:
        obj: Request dict (model, messages, temperature, etc.)

    Returns:
        64-character hex string
    """
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf8")
    ).hexdigest()



def _convert_tools_for_openai(tools: list) -> list:
    """
    Convert Tool objects to OpenAI's tool calling format.

    Args:
        tools: List of airt.Tool objects

    Returns:
        List of dicts in OpenAI's {"type": "function", "function": {}} format
    """
    payload = []
    for tool in tools:

        if tool.input_schema is not None:
            payload.append({
                "type": "function",
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.input_schema.model_json_schema(),
            })

        else:
            # When there is no implementation, there is no function to call,
            # and instead we just use the tool's name to trigger the native capability
            payload.append({"type": tool.name})

    return payload


def _convert_tools_for_gemini(tools: list) -> list:
    """
    Convert Tool objects to Gemini's FunctionDeclaration format.

    Args:
        tools: List of airt.Tool objects

    Returns:
        List containing single types.Tool with all function declarations
    """
    fn_decls = []
    payload = []
    for tool in tools:

        if tool.input_schema is not None:

            fn_decls.append(
                types.FunctionDeclaration(
                    name=tool.name,
                    description=tool.description,
                    parameters=tool.input_schema.model_json_schema(),
                )
            )

        elif tool.name == 'web_search':
            payload.append(types.Tool(google_search=types.GoogleSearch()))
        else:
            raise ValueError(f"Tool '{tool.name}' does not have an implementation, a schema, or is not supported.")

    if fn_decls:
        payload.extend([types.Tool(function_declarations=fn_decls)])
    return payload


def _convert_tools_for_claude(tools: list) -> list:
    """
    Convert Tool objects to Claude's tool format.

    Args:
        tools: List of airt.Tool objects

    Returns:
        List of dicts with name, description, and input_schema keys
    """

    payload = []
    for tool in tools:

        if tool.name == 'web_search':
            raise NotImplementedError("web_search is not supported for Claude yet.")

        payload.append(
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.input_schema.model_json_schema(),
            }
        )
    return payload


def query_llm(
    system_prompt: str = "",
    user_inputs: Optional[List[str]] = None,
    *,
    cache: Optional[str] = "llm_cache",
    api_key: Optional[str] = None,
    model: str = "gpt-4.1",
    temperature: float = 0.7,
    max_tokens: int = 4000,
    max_parallel_calls: int = 20,
    tools: Optional[list] = None,
    require_tool: bool = False
) -> Union[str, List[str], dict, List[dict]]:
    """
    Unified LLM query function with automatic provider routing and caching.

    Routes to OpenAI, Gemini, or Claude based on model name prefix. Supports
    batching, async execution, disk caching, and tool calling across all providers.

    Args:
        system_prompt: System/instruction message (placement varies by provider)
        user_inputs: List of user messages (None/empty = single call with system prompt only)
        cache: Directory name for diskcache (None = no caching)
        api_key: Provider API key (falls back to env vars)
        model: Model identifier (e.g., "gpt-4", "gemini-2.0-flash-exp", "claude-3-5-sonnet")
        temperature: Sampling temperature (0.0 = deterministic, 1.0 = creative)
        max_tokens: Maximum output tokens
        max_parallel_calls: Concurrent request limit for batching
        tools: List of Tool objects for function calling
        require_tool: Force model to call a tool (no free-text response)

    Returns:
        - Single request: str (text) or dict (tool call)
        - Multiple requests: List[str] or List[dict]

    Raises:
        ValueError: Unknown model prefix
        RuntimeError: Missing API key or tool call failure

    Provider routing:
        - "gpt-*" or "o1-*" → OpenAI
        - "gemini-*" → Google Gemini
        - "claude-*" → Anthropic Claude

    Caching behavior:
        - Cache key includes model, messages, temperature, tools
        - Identical requests return cached results instantly
        - Set cache=None to disable

    Tool calling:
        - tools=[] enables function calling
        - require_tool=True forces tool use (errors if model returns text)
        - Returns dict: {"tool_name": str, "arguments": dict}

    Examples:
         # Simple text generation
         query_llm("You are helpful.", ["What is 2+2?"], model="gpt-4")
        '4'

         # Batch multiple queries
         query_llm("", ["Question 1", "Question 2"], model="gemini-2.0-flash-exp")
        ['Answer 1', 'Answer 2']

         # Tool calling
         tools = [vector_search_tool]
         query_llm("", ["Find info"], model="claude-3-5-sonnet", tools=tools, require_tool=True)
        {'tool_name': 'vector_search', 'arguments': {'query': 'info', 'top_k': 5}}
    """

    if user_inputs is None:
        user_inputs = []

    # Route to provider based on model name prefix
    model_lower = model.lower()
    if model_lower.startswith(("gpt-", "o1-")):
        return _query_openai(
            system_prompt, user_inputs, cache, api_key, model,
            temperature, max_tokens, max_parallel_calls, tools,
            require_tool
        )
    elif model_lower.startswith("gemini-"):
        return _query_gemini(
            system_prompt, user_inputs, cache, api_key, model,
            temperature, max_tokens, max_parallel_calls, tools,
            require_tool
        )
    elif model_lower.startswith("claude-"):
        return _query_claude(
            system_prompt, user_inputs, cache, api_key, model,
            temperature, max_tokens, max_parallel_calls, tools,
            require_tool
        )

    else:
        raise ValueError(
            f"Unknown model provider for model '{model}'. "
            "Expected model name to start with 'gpt-', 'o1-', 'gemini-', or 'claude-'"
        )


def _query_openai(
    system_prompt: str,
    user_inputs: List[str],
    cache: Optional[str],
    api_key: Optional[str],
    model: str,
    temperature: float,
    max_tokens: int,
    max_parallel_calls: int,
    tools: Optional[list],
    require_tool: bool = False  # OpenAI doesn't support strict tool enforcement yet
) -> Union[str, List[str], dict, List[dict]]:
    """
    OpenAI-specific query implementation.

    Handles GPT-4, GPT-4o, o1, and other OpenAI models. Supports async batching
    and tool calling via chat.completions API.

    Args:
        See query_llm() for parameter descriptions

    Returns:
        Text string(s) or tool call dict(s)

    Note:
        OpenAI places system_prompt in separate message with role="system".
    """
    # Resolve API key from args or environment
    resolved_key = api_key or os.getenv("OPENAI_API_KEY")
    if not resolved_key:
        raise RuntimeError(
            "OpenAI API key not provided. "
            "Pass api_key= or set OPENAI_API_KEY."
        )

    cache_obj = diskcache.Cache(cache) if cache is not None else None
    client = openai.AsyncOpenAI(api_key=resolved_key)

    # Build request payloads
    requests: List[dict] = []

    def build_messages(user_text: Optional[str]) -> list[dict]:
        """Construct messages array for OpenAI chat API."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        if user_text is not None:
            messages.append({"role": "user", "content": user_text})
        return messages

    # Treat empty user_inputs as single request with system prompt only
    if not user_inputs:
        user_inputs = [None]

    for user_text in user_inputs:
        req = {
            "model": model,
            "input": build_messages(user_text),
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if tools:
            req["tools"] = _convert_tools_for_openai(tools)
        requests.append(req)

    # Semaphore limits concurrent requests
    semaphore = asyncio.Semaphore(max_parallel_calls)

    async def _one_call(req: dict):
        """Execute single cached/async request."""
        key = _hash_request(req)

        # Check cache first
        if cache_obj is not None and key in cache_obj:
            return cache_obj[key]
        async with semaphore:

            resp = await client.responses.create(**req)
            text_parts = []
            tool_call = None

            for item in resp.output:

                # Case 1: native web search call (you can ignore or log it)
                if item.type == "web_search_call":
                    continue

                # Case 2: assistant message
                if item.type == "message":
                    for part in item.content:
                        if part.type == "output_text":
                            text_parts.append(part.text)

                # Case 3: function tool call
                if item.type == "function_call":
                    tool_call = item

            if tools and tool_call:

                args = tool_call.arguments
                if isinstance(args, str):
                    args = json.loads(args)

                result = {
                    "tool_name": tool_call.name,
                    "arguments": args or {},
                }
            else:
                result = "".join(text_parts)

            # Cache result
            if cache_obj is not None:
                cache_obj[key] = result

            return result

    async def _run_all():
        """Batch all requests concurrently."""
        return await asyncio.gather(*(_one_call(r) for r in requests))

    outputs = asyncio.run(_run_all())

    # Return scalar for single request, list for multiple
    if len(outputs) == 1:
        return outputs[0]
    return outputs


def _query_gemini(
    system_prompt: str,
    user_inputs: List[str],
    cache: Optional[str],
    api_key: Optional[str],
    model: str,
    temperature: float,
    max_tokens: int,
    max_parallel_calls: int,
    tools: Optional[list],
    require_tool: bool = False
):
    """
    Gemini-specific query implementation.

    Handles gemini-1.5-pro, gemini-2.0-flash-exp, and other Google models.
    Supports tool calling with mode="ANY" for strict enforcement.

    Args:
        See query_llm() for parameter descriptions

    Returns:
        Text string(s) or tool call dict(s)

    Note:
        Gemini concatenates system_prompt and user_text into single content string.
        Supports require_tool via mode="ANY" in FunctionCallingConfig.
    """
    # Resolve API key (tries GEMINI_API_KEY, then GOOGLE_API_KEY)
    resolved_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not resolved_key:
        raise RuntimeError("Gemini API key not provided")

    client = genai.Client(api_key=resolved_key)
    cache_obj = diskcache.Cache(cache) if cache is not None else None
    semaphore = asyncio.Semaphore(max_parallel_calls)

    def build_text(user_text: Optional[str]) -> str:
        """Combine system and user prompts into single string."""
        if system_prompt and user_text:
            return f"{system_prompt}\n\n{user_text}"
        return system_prompt or user_text or ""

    if not user_inputs:
        user_inputs = [None]


    # This part is extremely ugly. I don't know yet whats a better way to solve this.
    # Gemini distinguishes between native tools like GoogleSearch, which don't have function declarations,
    # and external tools, which do. I found no way to unify them yet. And config is meant to
    # enable function calls. So we have to make sure we create a config iff there are external
    # tools.
    gemini_tools = _convert_tools_for_gemini(tools) if tools else None

    has_function_decls = any(
        getattr(t, "function_declarations", None)
        for t in gemini_tools
    )

    # Configure tool calling mode
    if has_function_decls:

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            tools=gemini_tools,
            tool_config=types.ToolConfig(
                function_calling_config=types.FunctionCallingConfig(
                    mode="ANY" if require_tool else "AUTO"  # ANY = force tool call
                )
            ),
        )

    else:
        config = None

    def extract_tool_call(resp):
        """
        Extract tool call from Gemini response.

        Gemini can place function calls in multiple locations depending on
        API version and response structure. This tries both paths.

        Args:
            resp: Gemini GenerateContentResponse

        Returns:
            Dict with tool_name and arguments

        Raises:
            RuntimeError: No tool call found in response
        """
        # Path 1: top-level function_calls attribute
        fcs = getattr(resp, "function_calls", None)
        if fcs:
            for fc in fcs:
                if fc is not None and getattr(fc, "name", None):
                    return {
                        "tool_name": fc.name,
                        "arguments": dict(fc.args or {}),
                    }

        # Path 2: candidates/content/parts/function_call (nested structure)
        for cand in getattr(resp, "candidates", []) or []:
            content = getattr(cand, "content", None)
            if not content:
                continue

            for part in getattr(content, "parts", []) or []:
                fc = getattr(part, "function_call", None)
                if fc and getattr(fc, "name", None):
                    return {
                        "tool_name": fc.name,
                        "arguments": dict(fc.args or {}),
                    }

        raise RuntimeError(
            "Gemini did not return a tool call.\n"
            f"Raw response:\n{resp}"
        )

    async def _one_call(user_text):
        """Execute single cached/async Gemini request."""
        # Build cache key from request parameters
        payload = {
            "model": model,
            "text": build_text(user_text),
            "tools": bool(tools),
            "require_tool": bool(require_tool),
            "tool_names": [t.name for t in (tools or [])],
            "tool_schemas": [t.input_schema.model_json_schema() if t.input_schema is not None else '' for t in (tools or [])],
        }

        key = _hash_request(payload)

        # Check cache
        if cache_obj is not None and key in cache_obj:
            return cache_obj[key]

        async with semaphore:
            # Gemini SDK is sync, so wrap in thread
            resp = await asyncio.to_thread(
                client.models.generate_content,
                model=model,
                contents=build_text(user_text),
                config=config,
            )

            # Parse response based on tool mode
            if tools and require_tool:
                result = extract_tool_call(resp)
            elif tools:
                # Tools available but optional
                if resp.function_calls:
                    result = extract_tool_call(resp)
                else:
                    result = resp.text
            else:
                result = resp.text

            # Cache result
            if cache_obj is not None:
                cache_obj[key] = result

            return result

    async def _run_all():
        """Batch all requests concurrently."""
        return await asyncio.gather(*(_one_call(u) for u in user_inputs))

    outputs = asyncio.run(_run_all())
    return outputs[0] if len(outputs) == 1 else outputs


def _query_claude(
    system_prompt: str,
    user_inputs: List[str],
    cache: Optional[str],
    api_key: Optional[str],
    model: str,
    temperature: float,
    max_tokens: int,
    max_parallel_calls: int,
    tools: Optional[list],
    require_tool: bool = False,
) -> Union[str, List[str], dict, List[dict]]:
    """Claude-specific implementation with parity vs OpenAI/Gemini tool semantics."""

    # ---- resolve API key ----
    resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY")
    if not resolved_key:
        raise RuntimeError(
            "Claude API key not provided. "
            "Pass api_key= or set ANTHROPIC_API_KEY."
        )

    client = anthropic.AsyncAnthropic(api_key=resolved_key)
    cache_obj = diskcache.Cache(cache) if cache is not None else None
    semaphore = asyncio.Semaphore(max_parallel_calls)

    if not user_inputs:
        user_inputs = [None]

    claude_tools = _convert_tools_for_claude(tools) if tools else None

    def build_messages(user_text: Optional[str]) -> list[dict]:
        # Claude requires at least one message
        if not user_text:
            return [{"role": "user", "content": "."}] # Messages must be non-empty
        return [{"role": "user", "content": user_text}]

    def extract_text(resp) -> str:
        parts = []
        for block in getattr(resp, "content", []) or []:
            if getattr(block, "type", None) == "text":
                parts.append(getattr(block, "text", ""))
        return "".join(parts)

    def extract_tool_use(resp) -> Optional[dict]:
        for block in getattr(resp, "content", []) or []:
            if getattr(block, "type", None) == "tool_use":
                return {
                    "tool_name": getattr(block, "name", None),
                    "arguments": getattr(block, "input", None) or {},
                    # Optional but useful later if you implement tool_result loops:
                    # "tool_use_id": getattr(block, "id", None),
                }
        return None

    async def _one_call(user_text: Optional[str]):
        payload: dict = {
            "model": model,
            "messages": build_messages(user_text),
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if system_prompt:
            payload["system"] = system_prompt

        if claude_tools:
            payload["tools"] = claude_tools
            payload["tool_choice"] = {"type": "any"} if require_tool else {"type": "auto"}

        key = _hash_request(payload)
        if cache_obj is not None and key in cache_obj:
            return cache_obj[key]

        async with semaphore:
            resp = await client.messages.create(**payload)

            if claude_tools:
                tool_call = extract_tool_use(resp)
                if tool_call is not None:
                    result = tool_call
                else:
                    if require_tool:
                        raise RuntimeError("Claude did not return a tool call")
                    result = extract_text(resp)
            else:
                result = extract_text(resp)

            if cache_obj is not None:
                cache_obj[key] = result

            return result

    async def _run_all():
        return await asyncio.gather(*(_one_call(u) for u in user_inputs))

    outputs = asyncio.run(_run_all())
    return outputs[0] if len(outputs) == 1 else outputs