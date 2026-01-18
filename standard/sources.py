"""
the hierarchy is:

invoke>_call>_take_next_step>_consume_next_step>_iter_next_step>_prepare_intermediate_steps

or

AgentExecutor.invoke()
└─ _call()                      ← agent main loop
   └─ while should_continue:
      └─ _take_next_step()
         └─ list(_iter_next_step())
            ├─ _prepare_intermediate_steps()
            ├─ agent.plan(...)  ← THE ONLY PLACE THE LLM IS CALLED
            ├─ yield AgentAction(s)
            └─ yield AgentStep(s) (tool execution)
         └─ _consume_next_step()


The steps are actually created by next_step_output = self._take_next_step(...) which eventually goes to .plan() which eventually use the prompt template

while not finished:
    # 1. Ask the LLM what to do
    decision = plan(prompt(intermediate_steps))

    # 2. If finished → return
    if decision is AgentFinish:
        return

    # 3. Otherwise execute tool(s)
    for action in decision:
        observation = run_tool(action)
        intermediate_steps.append((action, observation))


"""



def _prepare_intermediate_steps(
        self,
        intermediate_steps: list[tuple[AgentAction, str]],
) -> list[tuple[AgentAction, str]]:
    """Name: langchain_classic.agents.agent.AgentExecutor._prepare_intermediate_steps
    Class: langchain_classic.agents.agent.AgentExecutor
    File: C:\Users\Yugen\AppData\Roaming\Python\Python314\site-packages\langchain_classic\agents\agent.py"""

    if (
            isinstance(self.trim_intermediate_steps, int)
            and self.trim_intermediate_steps > 0
    ):
        return intermediate_steps[-self.trim_intermediate_steps :]
    if callable(self.trim_intermediate_steps):
        return self.trim_intermediate_steps(intermediate_steps)
    return intermediate_steps


"""Name: langchain_classic.agents.agent.AgentExecutor._iter_next_step
Class: langchain_classic.agents.agent.AgentExecutor
File: C:\Users\Yugen\AppData\Roaming\Python\Python314\site-packages\langchain_classic\agents\agent.py
"""
def _iter_next_step(
    self,
    name_to_tool_map: dict[str, BaseTool],
    color_mapping: dict[str, str],
    inputs: dict[str, str],
    intermediate_steps: list[tuple[AgentAction, str]],
    run_manager: CallbackManagerForChainRun | None = None,
) -> Iterator[AgentFinish | AgentAction | AgentStep]:
    """Take a single step in the thought-action-observation loop.

    Override this to take control of how the agent makes and acts on choices.
    """
    try:
        intermediate_steps = self._prepare_intermediate_steps(intermediate_steps)

        # Call the LLM to see what to do.
        output = self._action_agent.plan(
            intermediate_steps,
            callbacks=run_manager.get_child() if run_manager else None,
            **inputs,
        )
    except OutputParserException as e:
        if isinstance(self.handle_parsing_errors, bool):
            raise_error = not self.handle_parsing_errors
        else:
            raise_error = False
        if raise_error:
            msg = (
                "An output parsing error occurred. "
                "In order to pass this error back to the agent and have it try "
                "again, pass `handle_parsing_errors=True` to the AgentExecutor. "
                f"This is the error: {e!s}"
            )
            raise ValueError(msg) from e
        text = str(e)
        if isinstance(self.handle_parsing_errors, bool):
            if e.send_to_llm:
                observation = str(e.observation)
                text = str(e.llm_output)
            else:
                observation = "Invalid or incomplete response"
        elif isinstance(self.handle_parsing_errors, str):
            observation = self.handle_parsing_errors
        elif callable(self.handle_parsing_errors):
            observation = self.handle_parsing_errors(e)
        else:
            msg = "Got unexpected type of `handle_parsing_errors`"  # type: ignore[unreachable]
            raise ValueError(msg) from e  # noqa: TRY004
        output = AgentAction("_Exception", observation, text)
        if run_manager:
            run_manager.on_agent_action(output, color="green")
        tool_run_kwargs = self._action_agent.tool_run_logging_kwargs()
        observation = ExceptionTool().run(
            output.tool_input,
            verbose=self.verbose,
            color=None,
            callbacks=run_manager.get_child() if run_manager else None,
            **tool_run_kwargs,
        )
        yield AgentStep(action=output, observation=observation)
        return

    # If the tool chosen is the finishing tool, then we end and return.
    if isinstance(output, AgentFinish):
        yield output
        return

    actions: list[AgentAction]
    actions = [output] if isinstance(output, AgentAction) else output
    for agent_action in actions:
        yield agent_action
    for agent_action in actions:
        yield self._perform_agent_action(
            name_to_tool_map,
            color_mapping,
            agent_action,
            run_manager,
        )


def _call(
        self,
        inputs: dict[str, str],
        run_manager: CallbackManagerForChainRun | None = None,
) -> dict[str, Any]:
    """Run text through and get agent response."""
    # Construct a mapping of tool name to tool for easy lookup
    name_to_tool_map = {tool.name: tool for tool in self.tools}
    # We construct a mapping from each tool to a color, used for logging.
    color_mapping = get_color_mapping(
        [tool.name for tool in self.tools],
        excluded_colors=["green", "red"],
    )
    intermediate_steps: list[tuple[AgentAction, str]] = []
    # Let's start tracking the number of iterations and time elapsed
    iterations = 0
    time_elapsed = 0.0
    start_time = time.time()
    # We now enter the agent loop (until it returns something).
    while self._should_continue(iterations, time_elapsed):
        next_step_output = self._take_next_step(
            name_to_tool_map,
            color_mapping,
            inputs,
            intermediate_steps,
            run_manager=run_manager,
        )
        if isinstance(next_step_output, AgentFinish):
            return self._return(
                next_step_output,
                intermediate_steps,
                run_manager=run_manager,
            )

        intermediate_steps.extend(next_step_output)
        if len(next_step_output) == 1:
            next_step_action = next_step_output[0]
            # See if tool should return directly
            tool_return = self._get_tool_return(next_step_action)
            if tool_return is not None:
                return self._return(
                    tool_return,
                    intermediate_steps,
                    run_manager=run_manager,
                )
        iterations += 1
        time_elapsed = time.time() - start_time
    output = self._action_agent.return_stopped_response(
        self.early_stopping_method,
        intermediate_steps,
        **inputs,
    )
    return self._return(output, intermediate_steps, run_manager=run_manager)


def _take_next_step(
    self,
    name_to_tool_map: dict[str, BaseTool],
    color_mapping: dict[str, str],
    inputs: dict[str, str],
    intermediate_steps: list[tuple[AgentAction, str]],
    run_manager: CallbackManagerForChainRun | None = None,
) -> AgentFinish | list[tuple[AgentAction, str]]:
    return self._consume_next_step(
        list(
            self._iter_next_step(
                name_to_tool_map,
                color_mapping,
                inputs,
                intermediate_steps,
                run_manager,
            ),
        ),
    )

def _consume_next_step(
    self,
    values: NextStepOutput,
) -> AgentFinish | list[tuple[AgentAction, str]]:
    if isinstance(values[-1], AgentFinish):
        if len(values) != 1:
            msg = "Expected a single AgentFinish output, but got multiple values."
            raise ValueError(msg)
        return values[-1]
    return [(a.action, a.observation) for a in values if isinstance(a, AgentStep)]



def _iter_next_step(
        self,
        name_to_tool_map: dict[str, BaseTool],
        color_mapping: dict[str, str],
        inputs: dict[str, str],
        intermediate_steps: list[tuple[AgentAction, str]],
        run_manager: CallbackManagerForChainRun | None = None,
) -> Iterator[AgentFinish | AgentAction | AgentStep]:
    """Take a single step in the thought-action-observation loop.

    Override this to take control of how the agent makes and acts on choices.
    """
    try:
        intermediate_steps = self._prepare_intermediate_steps(intermediate_steps)

        # Call the LLM to see what to do.
        output = self._action_agent.plan(
            intermediate_steps,
            callbacks=run_manager.get_child() if run_manager else None,
            **inputs,
        )
    except OutputParserException as e:
        if isinstance(self.handle_parsing_errors, bool):
            raise_error = not self.handle_parsing_errors
        else:
            raise_error = False
        if raise_error:
            msg = (
                "An output parsing error occurred. "
                "In order to pass this error back to the agent and have it try "
                "again, pass `handle_parsing_errors=True` to the AgentExecutor. "
                f"This is the error: {e!s}"
            )
            raise ValueError(msg) from e
        text = str(e)
        if isinstance(self.handle_parsing_errors, bool):
            if e.send_to_llm:
                observation = str(e.observation)
                text = str(e.llm_output)
            else:
                observation = "Invalid or incomplete response"
        elif isinstance(self.handle_parsing_errors, str):
            observation = self.handle_parsing_errors
        elif callable(self.handle_parsing_errors):
            observation = self.handle_parsing_errors(e)
        else:
            msg = "Got unexpected type of `handle_parsing_errors`"  # type: ignore[unreachable]
            raise ValueError(msg) from e  # noqa: TRY004
        output = AgentAction("_Exception", observation, text)
        if run_manager:
            run_manager.on_agent_action(output, color="green")
        tool_run_kwargs = self._action_agent.tool_run_logging_kwargs()
        observation = ExceptionTool().run(
            output.tool_input,
            verbose=self.verbose,
            color=None,
            callbacks=run_manager.get_child() if run_manager else None,
            **tool_run_kwargs,
        )
        yield AgentStep(action=output, observation=observation)
        return

    # If the tool chosen is the finishing tool, then we end and return.
    if isinstance(output, AgentFinish):
        yield output
        return

    actions: list[AgentAction]
    actions = [output] if isinstance(output, AgentAction) else output
    for agent_action in actions:
        yield agent_action
    for agent_action in actions:
        yield self._perform_agent_action(
            name_to_tool_map,
            color_mapping,
            agent_action,
            run_manager,
        )


def _get_tool_return(
    self,
    next_step_output: tuple[AgentAction, str],
) -> AgentFinish | None:
    """Check if the tool is a returning tool."""
    agent_action, observation = next_step_output
    name_to_tool_map = {tool.name: tool for tool in self.tools}
    return_value_key = "output"
    if len(self._action_agent.return_values) > 0:
        return_value_key = self._action_agent.return_values[0]
    # Invalid tools won't be in the map, so we return False.
    if (
        agent_action.tool in name_to_tool_map
        and name_to_tool_map[agent_action.tool].return_direct
    ):
        return AgentFinish(
            {return_value_key: observation},
            "",
        )
    return None
