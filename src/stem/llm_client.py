# Unified LLM client. Supports OpenAI and Anthropic interchangeably.
# TOKEN BUDGET AWARENESS: stem phase uses ~2000 tokens per call.
# With 15 RL iterations × tree search branching factor 3 = 45 calls minimum.
# Total budget estimate: ~100k tokens for a full experiment run.
# This is relevant to the "fragile execution under cognitive load" failure
# archetype identified in KAMI benchmark (arXiv:2512.07497).
from __future__ import annotations

import json
import os
import time

import structlog

from src.stem.models import LLMMessage, LLMResponse

logger = structlog.get_logger(__name__)

DEFAULT_TEMPERATURE: float = 0.3
DEFAULT_MAX_TOKENS: int = 2000
LOCK_THRESHOLD: float = 0.8

_JSON_INSTRUCTION = "Respond with valid JSON only. No markdown, no explanation."
_JSON_RETRY_PREFIX = "Your previous response was not valid JSON. Try again.\n\n"


class LLMClientError(Exception):
    pass


class LLMClient:
    def __init__(self) -> None:
        self._provider = os.environ.get("MODEL_PROVIDER", "").lower()
        self._model_name = os.environ.get("MODEL_NAME", "")

        if not self._provider:
            raise LLMClientError("MODEL_PROVIDER environment variable is not set")
        if not self._model_name:
            raise LLMClientError("MODEL_NAME environment variable is not set")
        if self._provider not in ("openai", "anthropic"):
            raise LLMClientError(
                f"Unsupported MODEL_PROVIDER: {self._provider!r}. "
                "Must be 'openai' or 'anthropic'."
            )

        if self._provider == "openai":
            api_key = os.environ.get("OPENAI_API_KEY", "")
            if not api_key:
                raise LLMClientError("OPENAI_API_KEY environment variable is not set")
            try:
                import openai
                self._client = openai.OpenAI(api_key=api_key)
            except Exception as exc:
                raise LLMClientError(f"Failed to initialize OpenAI client: {exc}") from exc
        else:
            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                raise LLMClientError("ANTHROPIC_API_KEY environment variable is not set")
            try:
                import anthropic
                self._client = anthropic.Anthropic(api_key=api_key)
            except Exception as exc:
                raise LLMClientError(f"Failed to initialize Anthropic client: {exc}") from exc

    def complete(
        self,
        messages: list[LLMMessage],
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> LLMResponse:
        if not messages:
            raise LLMClientError("messages must not be empty")

        start = time.monotonic()
        try:
            if self._provider == "openai":
                response = self._complete_openai(messages, temperature, max_tokens)
            else:
                response = self._complete_anthropic(messages, temperature, max_tokens)
        except LLMClientError:
            raise
        except Exception as exc:
            raise LLMClientError(f"LLM API call failed: {exc}") from exc

        duration_ms = int((time.monotonic() - start) * 1000)
        logger.info(
            "llm_client.complete",
            model=response.model,
            input_tokens=response.input_tokens,
            output_tokens=response.output_tokens,
            duration_ms=duration_ms,
        )
        return response

    def complete_json(
        self,
        messages: list[LLMMessage],
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> dict:
        if not messages:
            raise LLMClientError("messages must not be empty")

        augmented = _inject_json_instruction(messages)
        response = self.complete(augmented, temperature, max_tokens)

        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            logger.warning("llm_client.json_parse_failed_retry", raw=response.content[:200])

        retry_messages = _prepend_retry(augmented, response.content)
        retry_response = self.complete(retry_messages, temperature, max_tokens)

        try:
            return json.loads(retry_response.content)
        except json.JSONDecodeError as exc:
            raise LLMClientError(
                f"LLM returned invalid JSON after retry: {retry_response.content[:200]}"
            ) from exc

    def _complete_openai(
        self,
        messages: list[LLMMessage],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        try:
            result = self._client.chat.completions.create(
                model=self._model_name,
                messages=[{"role": m.role, "content": m.content} for m in messages],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return LLMResponse(
                content=result.choices[0].message.content or "",
                model=result.model,
                input_tokens=result.usage.prompt_tokens,
                output_tokens=result.usage.completion_tokens,
                raw_response=result.model_dump(),
            )
        except Exception as exc:
            raise LLMClientError(f"OpenAI API error: {exc}") from exc

    def _complete_anthropic(
        self,
        messages: list[LLMMessage],
        temperature: float,
        max_tokens: int,
    ) -> LLMResponse:
        try:
            system_parts = [m.content for m in messages if m.role == "system"]
            convo = [
                {"role": m.role, "content": m.content}
                for m in messages
                if m.role != "system"
            ]
            kwargs: dict = {
                "model": self._model_name,
                "messages": convo,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if system_parts:
                kwargs["system"] = "\n\n".join(system_parts)

            result = self._client.messages.create(**kwargs)
            return LLMResponse(
                content=result.content[0].text,
                model=result.model,
                input_tokens=result.usage.input_tokens,
                output_tokens=result.usage.output_tokens,
                raw_response=result.model_dump(),
            )
        except Exception as exc:
            raise LLMClientError(f"Anthropic API error: {exc}") from exc


def _inject_json_instruction(messages: list[LLMMessage]) -> list[LLMMessage]:
    last = messages[-1]
    if _JSON_INSTRUCTION in last.content:
        return list(messages)
    updated_last = LLMMessage(role=last.role, content=f"{last.content}\n\n{_JSON_INSTRUCTION}")
    return list(messages[:-1]) + [updated_last]


def _prepend_retry(messages: list[LLMMessage], bad_content: str) -> list[LLMMessage]:
    retry_user = LLMMessage(
        role="user",
        content=f"{_JSON_RETRY_PREFIX}{bad_content}",
    )
    return list(messages) + [retry_user]
