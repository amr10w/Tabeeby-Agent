import json
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

try:
    from stores.llm.LLMInterface import LLMInterface
    from stores.llm.LLMEnums import OllamaEnums
    from prompts.prompt_templatet import VEZEETA_SYSTEM_PROMPT
except ImportError:
    from ..stores.llm.LLMInterface import LLMInterface
    from ..stores.llm.LLMEnums import OllamaEnums
    from ..prompts.prompt_templatet import VEZEETA_SYSTEM_PROMPT





class Agent:
    """Production-ready Autonomous Agent implementing a bounded reasoning loop

    (Send -> Check -> Append -> Run -> Append -> Repeat) with tool-calling support
    and medical guardrails for Tabeeby (Vezeeta Doctor-Finder).
    """

    def __init__(
        self,
        client: LLMInterface,
        tools: Optional[Sequence[Callable[..., Any]]] = None,
        max_iterations: int = 10,
        system_prompt: Optional[str] = None,
    ):
        """Initialize the Agent.

        Args:
            client: LLM client implementing LLMInterface.
            tools: Optional sequence of callable tool functions.
            max_iterations: Maximum reasoning loop iterations before fallback.
            system_prompt: Custom system prompt (defaults to VEZEETA_SYSTEM_PROMPT).
        """
        self.client = client
        self.tools: List[Callable[..., Any]] = list(tools) if tools else []
        self.tools_map: Dict[str, Callable[..., Any]] = {
            getattr(tool, "__name__", str(tool)): tool for tool in self.tools
        }
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt or VEZEETA_SYSTEM_PROMPT
        self.logger = logging.getLogger(self.__class__.__name__)

    def _construct_message(self, prompt: str, role: str, truncate: bool = False) -> Dict[str, Any]:
        """Construct a message dictionary using the client provider's construct_prompt method.

        Args:
            prompt: Text content of the message.
            role: Role enum value (e.g. system, user, assistant, tool).
            truncate: Whether to truncate text based on provider configuration.

        Returns:
            Dictionary representing the message for the LLM chat history.
        """
        if hasattr(self.client, "construct_prompt") and callable(self.client.construct_prompt):
            return self.client.construct_prompt(prompt=prompt, role=role, truncate=truncate)
        return {
            "role": role,
            "content": prompt,
        }

    def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> str:
        """Safely execute a tool by name with keyword arguments.

        Catches all runtime errors and returns them as formatted strings so the
        LLM can inspect errors and self-correct.

        Args:
            tool_name: Name of the tool to execute.
            tool_args: Dictionary of keyword arguments for the tool.

        Returns:
            Output converted to a string or JSON string.
        """
        tool = self.tools_map.get(tool_name)
        if not tool:
            available = list(self.tools_map.keys())
            error_msg = f"Error: Tool '{tool_name}' not found. Available tools: {available}"
            self.logger.warning(error_msg)
            return json.dumps({"error": error_msg})

        try:
            self.logger.info(f"Executing tool '{tool_name}' with arguments: {tool_args}")
            result = tool(**tool_args)

            if isinstance(result, str):
                return result
            elif isinstance(result, (dict, list, int, float, bool)):
                return json.dumps(result, ensure_ascii=False, default=str)
            elif result is None:
                return json.dumps({"status": "success", "result": None})
            elif hasattr(result, "model_dump"):
                # Pydantic v2
                return json.dumps(result.model_dump(), ensure_ascii=False, default=str)
            elif hasattr(result, "dict"):
                # Pydantic v1
                return json.dumps(result.dict(), ensure_ascii=False, default=str)
            else:
                return str(result)

        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}': {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return json.dumps({"error": error_msg})

    def _extract_tool_call_info(self, tool_call: Any) -> Tuple[str, Dict[str, Any]]:
        """Safely extract function name and argument dictionary from a tool call object or dict.

        Args:
            tool_call: ToolCall object or dictionary from LLM response.

        Returns:
            Tuple of (tool_name, tool_arguments_dict).
        """
        tool_name = ""
        raw_args: Any = {}

        if hasattr(tool_call, "function"):
            func = tool_call.function
            tool_name = getattr(func, "name", "")
            raw_args = getattr(func, "arguments", {})
        elif isinstance(tool_call, dict):
            func = tool_call.get("function", {})
            if isinstance(func, dict):
                tool_name = func.get("name", "")
                raw_args = func.get("arguments", {})
            else:
                tool_name = tool_call.get("name", "")
                raw_args = tool_call.get("arguments", {})
        else:
            tool_name = getattr(tool_call, "name", "")
            raw_args = getattr(tool_call, "arguments", {})

        if isinstance(raw_args, dict):
            tool_args = raw_args
        elif isinstance(raw_args, str):
            try:
                parsed = json.loads(raw_args)
                tool_args = parsed if isinstance(parsed, dict) else {"input": parsed}
            except (json.JSONDecodeError, TypeError):
                tool_args = {}
        elif raw_args is None:
            tool_args = {}
        else:
            tool_args = {"input": raw_args}

        return tool_name, tool_args

    def run(
        self,
        prompt: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
    ) -> Tuple[str, List[Dict[str, Any]], float]:
        """Execute the 6-step bounded reasoning loop.

        Workflow:
        1. Send: Submit accumulated context to the LLM via client.generate_response.
        2. Check: Determine if the response is a final answer or requests tool call(s).
        3. Append: Record the assistant's message into conversation history.
        4. Run: Execute requested tool(s) safely.
        5. Append: Record tool observations into conversation history via _construct_message.
        6. Repeat: Continue until a final response is generated or max_iterations is reached.

        Args:
            prompt: Incoming user prompt.
            chat_history: Previous conversation message history.
            system_prompt: Optional override for the system prompt.

        Returns:
            Tuple of (final_response_text, full_message_history, total_cost).
        """
        total_cost: float = 0.0
        active_system_prompt = system_prompt or self.system_prompt

        # Build conversation history: system prompt -> chat_history -> user prompt
        messages: List[Dict[str, Any]] = []

        if active_system_prompt:
            messages.append(
                self._construct_message(
                    prompt=active_system_prompt,
                    role=OllamaEnums.SYSTEM.value,
                    truncate=False,
                )
            )

        if chat_history:
            for msg in chat_history:
                # Avoid duplicating system message if already set at the root
                if (
                    msg.get("role") == OllamaEnums.SYSTEM.value
                    and messages
                    and messages[0].get("role") == OllamaEnums.SYSTEM.value
                ):
                    messages[0] = dict(msg)
                else:
                    messages.append(dict(msg))

        if prompt and prompt.strip():
            messages.append(
                self._construct_message(
                    prompt=prompt.strip(),
                    role=OllamaEnums.USER.value,
                    truncate=True,
                )
            )

        # Bounded Reasoning Loop
        for iteration in range(1, self.max_iterations + 1):
            self.logger.debug(f"Reasoning loop iteration {iteration}/{self.max_iterations}")

            # Step 1: Send messages to LLM
            response = self.client.generate_response(
                prompt="",
                chat_history=messages,
                tools=self.tools if self.tools else None,
            )

            # Safety check: Validate client response
            if not response or not hasattr(response, "message") or response.message is None:
                self.logger.error("LLM returned an empty or invalid response.")
                fallback_error = (
                    "I apologize, but I encountered an error communicating with the AI service. "
                    "Please try again."
                )
                messages.append(
                    self._construct_message(
                        prompt=fallback_error,
                        role=OllamaEnums.ASSISTANT.value,
                        truncate=False,
                    )
                )
                return fallback_error, messages, total_cost

            message_obj = response.message
            tool_calls = getattr(message_obj, "tool_calls", None)

            # Step 2: Check for exit condition (no tool calls -> final answer)
            if not tool_calls:
                final_content = getattr(message_obj, "content", "") or ""
                messages.append(
                    self._construct_message(
                        prompt=final_content,
                        role=OllamaEnums.ASSISTANT.value,
                        truncate=False,
                    )
                )
                return final_content, messages, total_cost

            # Step 3: Append assistant message with tool calls to history
            assistant_entry = self._construct_message(
                prompt=getattr(message_obj, "content", "") or "",
                role=OllamaEnums.ASSISTANT.value,
                truncate=False,
            )
            assistant_entry["tool_calls"] = tool_calls
            messages.append(assistant_entry)

            # Step 4 & 5: Run tools and Append observations
            for tool_call in tool_calls:
                tool_name, tool_args = self._extract_tool_call_info(tool_call)
                observation = self._execute_tool(tool_name, tool_args)

                tool_entry = self._construct_message(
                    prompt=observation,
                    role=OllamaEnums.TOOL.value,
                    truncate=False,
                )
                messages.append(tool_entry)

            # Step 6: Repeat loop with accumulated observations

        # Max iterations reached without final response
        self.logger.warning(
            f"Agent reached maximum iterations ({self.max_iterations}) without concluding."
        )
        fallback_msg = (
            "I apologize, but I was unable to complete your request within the allowed steps. "
            "Please try simplifying your request or asking a more specific question."
        )
        messages.append(
            self._construct_message(
                prompt=fallback_msg,
                role=OllamaEnums.ASSISTANT.value,
                truncate=False,
            )
        )
        return fallback_msg, messages, total_cost
