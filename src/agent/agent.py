"""Tabeeby Medical Agent Implementation.

Autonomous Agent implementing a bounded reasoning loop with tool-calling support,
medical safety guardrails, and RAG prompt formatting for Vezeeta Doctor-Finder.
"""

from __future__ import annotations

import inspect
import json
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

try:
    from prompts.prompt_templatet import (
        VEZEETA_SYSTEM_PROMPT,
        PROMPT_TEMPLATE,
        format_prompt,
    )
    from stores.llm.LLMEnums import OllamaEnums
    from stores.llm.LLMInterface import LLMInterface
    from stores.vectordb.VectorDBInterface import VectorDBInterface
except ImportError:
    from ..prompts.prompt_templatet import (
        VEZEETA_SYSTEM_PROMPT,
        PROMPT_TEMPLATE,
        format_prompt,
    )
    from ..stores.llm.LLMEnums import OllamaEnums
    from ..stores.llm.LLMInterface import LLMInterface
    from ..stores.vectordb.VectorDBInterface import VectorDBInterface


class Agent:
    """Production-ready Autonomous Medical Agent for Tabeeby (Vezeeta Doctor-Finder).

    Executes a bounded reasoning loop (Send -> Check -> Append -> Run -> Append -> Repeat)
    with tool-calling capabilities, RAG document grounding, and token cost tracking.
    """

    def __init__(
        self,
        client: LLMInterface,
        tools: Optional[Sequence[Callable[..., Any]]] = None,
        max_iterations: int = 10,
        system_prompt: Optional[str] = None,
        total_cost: float = 0.0,
        input_token_cost: float = 0.0000014,
        output_token_cost: float = 0.0000044,
    ):
        """Initialize the Agent.

        Args:
            client: Generation LLM provider implementing LLMInterface.
            tools: Optional sequence of tool callables or tool instances.
            max_iterations: Maximum reasoning loop iterations before fallback.
            system_prompt: Custom system prompt (defaults to VEZEETA_SYSTEM_PROMPT).
            client_embedding: Optional embedding provider for tool injection.
            client_vectorDB: Optional vector database provider for tool injection.
            total_cost: Initial accumulated cost (defaults to 0.0).
            input_token_cost: Cost per input token (default 0.0000014 = $1.40 / 1M tokens).
            output_token_cost: Cost per output token (default 0.0000044 = $4.40 / 1M tokens).
        """
        self.client = client
        self.max_iterations = max_iterations
        self.system_prompt = system_prompt or VEZEETA_SYSTEM_PROMPT
        self.total_cost: float = float(total_cost)
        self.input_token_cost: float = float(input_token_cost)
        self.output_token_cost: float = float(output_token_cost)
        self.logger = logging.getLogger(self.__class__.__name__)

        self.tools: List[Callable[..., Any]] = []
        if tools:
            for t in tools:
                if hasattr(t, "search_doctors") and callable(getattr(t, "search_doctors")):
                    self.tools.append(t.search_doctors)
                elif callable(t):
                    self.tools.append(t)

        self.tools_map: Dict[str, Callable[..., Any]] = {
            getattr(tool, "__name__", str(tool)): tool for tool in self.tools
        }

    def _construct_message(
        self, prompt: str, role: str, truncate: bool = False
    ) -> Dict[str, Any]:
        """Construct a chat message dictionary using provider conventions."""
        if hasattr(self.client, "construct_prompt") and callable(self.client.construct_prompt):
            return self.client.construct_prompt(prompt=prompt, role=role, truncate=truncate)
        return {
            "role": role,
            "content": prompt,
        }

    def _calculate_response_cost(self, response: Any) -> float:
        """Calculate the cost of an LLM response based on evaluated token counts.

        Extracts input and output tokens from Ollama response attributes
        (prompt_eval_count, eval_count) or standard usage dictionaries.
        """
        if not response:
            return 0.0

        prompt_tokens = 0
        completion_tokens = 0

        # Direct attributes (Ollama ChatResponse object)
        if hasattr(response, "prompt_eval_count") and response.prompt_eval_count is not None:
            prompt_tokens = response.prompt_eval_count
        elif isinstance(response, dict) and "prompt_eval_count" in response:
            prompt_tokens = response.get("prompt_eval_count") or 0

        if hasattr(response, "eval_count") and response.eval_count is not None:
            completion_tokens = response.eval_count
        elif isinstance(response, dict) and "eval_count" in response:
            completion_tokens = response.get("eval_count") or 0

        # Fallback to standard usage structure (OpenAI / Cloud APIs)
        usage = getattr(response, "usage", None) or (
            response.get("usage") if isinstance(response, dict) else None
        )
        if usage:
            if hasattr(usage, "prompt_tokens") and usage.prompt_tokens is not None:
                prompt_tokens = usage.prompt_tokens or prompt_tokens
            elif isinstance(usage, dict) and "prompt_tokens" in usage:
                prompt_tokens = usage.get("prompt_tokens") or prompt_tokens

            if hasattr(usage, "completion_tokens") and usage.completion_tokens is not None:
                completion_tokens = usage.completion_tokens or completion_tokens
            elif isinstance(usage, dict) and "completion_tokens" in usage:
                completion_tokens = usage.get("completion_tokens") or completion_tokens

        cost = (prompt_tokens * self.input_token_cost) + (
            completion_tokens * self.output_token_cost
        )
        return cost

    def _format_doctors_context(self, doctors: Any) -> str:
        """Format retrieved doctor records into structured context text for RAG."""
        if isinstance(doctors, str):
            try:
                doctors = json.loads(doctors)
            except Exception:
                return doctors

        if not isinstance(doctors, list) or len(doctors) == 0:
            return "No matching doctors found for the specified search criteria."

        formatted_docs = []
        for idx, doc in enumerate(doctors, 1):
            if not isinstance(doc, dict):
                formatted_docs.append(str(doc))
                continue

            lines = [f"Doctor {idx}:"]
            if doc.get("name"):
                lines.append(f"  - Name: {doc['name']}")
            if doc.get("specialty"):
                lines.append(f"  - Specialty: {doc['specialty']}")
            if doc.get("subspecialties_text"):
                lines.append(f"  - Subspecialties: {doc['subspecialties_text']}")
            if doc.get("address"):
                lines.append(f"  - Clinic Location / Area: {doc['address']}")
            if doc.get("fee") is not None:
                lines.append(f"  - Consultation Fee: {doc['fee']} EGP")
            if doc.get("reviews_count") is not None:
                lines.append(f"  - Patient Reviews: {doc['reviews_count']} reviews")
            if doc.get("waiting_time_min") is not None:
                lines.append(f"  - Average Waiting Time: ~{doc['waiting_time_min']} mins")
            if doc.get("profile_url"):
                lines.append(f"  - Profile & Booking URL: {doc['profile_url']}")
            formatted_docs.append("\n".join(lines))

        return "\n\n".join(formatted_docs)

    def _execute_tool(self, tool_name: str, tool_args: Dict[str, Any]) -> Tuple[Any, str]:
        """Safely execute a tool by name with keyword arguments.

        Returns:
            Tuple of (raw_result, string_observation).
        """
        tool = self.tools_map.get(tool_name)
        if not tool:
            available = list(self.tools_map.keys())
            error_msg = f"Error: Tool '{tool_name}' not found. Available tools: {available}"
            self.logger.warning(error_msg)
            return {"error": error_msg}, json.dumps({"error": error_msg})

        try:
            call_kwargs = dict(tool_args)
            try:
                sig = inspect.signature(tool)
                if (
                    "client_embedding" in sig.parameters
                    and "client_embedding" not in call_kwargs
                    and self.client_embedding
                ):
                    call_kwargs["client_embedding"] = self.client_embedding
                if (
                    "client_vectorDB" in sig.parameters
                    and "client_vectorDB" not in call_kwargs
                    and self.client_vectorDB
                ):
                    call_kwargs["client_vectorDB"] = self.client_vectorDB
            except Exception as sig_err:
                self.logger.debug(f"Signature inspection skipped for '{tool_name}': {sig_err}")

            self.logger.info(f"Executing tool '{tool_name}' with arguments: {tool_args}")
            raw_result = tool(**call_kwargs)

            if isinstance(raw_result, str):
                str_res = raw_result
            elif isinstance(raw_result, (dict, list, int, float, bool)):
                str_res = json.dumps(raw_result, ensure_ascii=False, default=str)
            elif raw_result is None:
                str_res = json.dumps({"status": "success", "result": None})
            elif hasattr(raw_result, "model_dump"):
                str_res = json.dumps(raw_result.model_dump(), ensure_ascii=False, default=str)
            elif hasattr(raw_result, "dict"):
                str_res = json.dumps(raw_result.dict(), ensure_ascii=False, default=str)
            else:
                str_res = str(raw_result)

            return raw_result, str_res

        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}': {str(e)}"
            self.logger.error(error_msg, exc_info=True)
            return {"error": error_msg}, json.dumps({"error": error_msg})

    def _extract_tool_call_info(self, tool_call: Any) -> Tuple[str, Dict[str, Any]]:
        """Safely extract function name and argument dictionary from a tool call object."""
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

    def reset_total_cost(self, new_cost: float = 0.0) -> None:
        """Reset or set the accumulated total cost on the agent instance."""
        self.total_cost = float(new_cost)

    def run(
        self,
        prompt: str,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        system_prompt: Optional[str] = None,
        total_cost: Optional[float] = None,
    ) -> Tuple[str, List[Dict[str, Any]], float]:
        """Execute the bounded reasoning loop for a patient inquiry.

        Workflow:
        1. Context setup: system prompt -> prior conversation history -> patient prompt.
        2. Bounded reasoning iterations:
           - Send context to LLM with tool definitions.
           - Calculate token cost from response and add to total cost.
           - If no tool calls -> extract final text and finish.
           - If tool calls -> execute tools, record tool observations, and format RAG context.
        3. Repeat until final answer or max_iterations is reached.

        Args:
            prompt: Patient's incoming query or symptom description.
            chat_history: Prior conversation message history.
            system_prompt: Optional system prompt override.
            total_cost: Optional starting total cost (e.g. from previous runs). If omitted,
                uses the agent instance's cumulative cost tracker.

        Returns:
            Tuple of (final_response_text, message_history, total_cost).
        """
        running_cost: float = (
            float(total_cost) if total_cost is not None else self.total_cost
        )
        active_system_prompt = system_prompt or self.system_prompt

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

        # Reasoning loop
        for iteration in range(1, self.max_iterations + 1):
            self.logger.debug(f"Reasoning loop iteration {iteration}/{self.max_iterations}")

            # Step 1: Query LLM
            response = self.client.generate_response(
                prompt=prompt,
                chat_history=messages,
                tools=self.tools if self.tools else None,
            )

            # Calculate cost of this LLM step
            step_cost = self._calculate_response_cost(response)
            running_cost += step_cost
            self.total_cost += step_cost

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
                return fallback_error, messages, running_cost

            message_obj = response.message
            tool_calls = getattr(message_obj, "tool_calls", None)

            # Step 2: Check for exit condition (no tool calls -> final response)
            if not tool_calls:
                final_content = getattr(message_obj, "content", "") or ""
                messages.append(
                    self._construct_message(
                        prompt=final_content,
                        role=OllamaEnums.ASSISTANT.value,
                        truncate=False,
                    )
                )
                return final_content, messages, running_cost

            # Step 3: Append assistant tool call message to history
            assistant_entry = self._construct_message(
                prompt=getattr(message_obj, "content", "") or "",
                role=OllamaEnums.ASSISTANT.value,
                truncate=False,
            )
            assistant_entry["tool_calls"] = tool_calls
            messages.append(assistant_entry)

            # Step 4 & 5: Run tools, record observations, and format RAG context
            for tool_call in tool_calls:
                tool_name, tool_args = self._extract_tool_call_info(tool_call)
                raw_result, observation_str = self._execute_tool(tool_name, tool_args)

                # Append standard tool observation message
                tool_entry = self._construct_message(
                    prompt=observation_str,
                    role=OllamaEnums.TOOL.value,
                    truncate=False,
                )
                messages.append(tool_entry)

                # If doctor search was performed, append formatted RAG template
                if "search" in tool_name.lower() or "doctor" in tool_name.lower():
                    context_str = self._format_doctors_context(raw_result)
                    rag_prompt = format_prompt(question=prompt, context=context_str)
                    rag_entry = self._construct_message(
                        prompt=rag_prompt,
                        role=OllamaEnums.USER.value,
                        truncate=False,
                    )
                    messages.append(rag_entry)

        # Max iterations reached
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
        return fallback_msg, messages, running_cost