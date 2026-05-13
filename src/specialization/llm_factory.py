from __future__ import annotations

import os

import structlog

from src.stem.llm_client import LLMClient

logger = structlog.get_logger(__name__)

MODEL_ROUTING: dict[str, str] = {
    "agent_review": os.getenv("MODEL_NAME_CHEAP", "gpt-4o-mini"),
    "gradient": os.getenv("MODEL_NAME_STRONG", "gpt-4o"),
    "variant_gen": os.getenv("MODEL_NAME_STRONG", "gpt-4o"),
    "ewc_merge": os.getenv("MODEL_NAME_STRONG", "gpt-4o"),
}

_VALID_KEYS = frozenset(MODEL_ROUTING.keys())


def get_llm_client(model_key: str) -> LLMClient:
    if model_key not in _VALID_KEYS:
        raise ValueError(
            f"Unknown model_key {model_key!r}. Valid keys: {sorted(_VALID_KEYS)}"
        )

    model_name = MODEL_ROUTING[model_key]
    provider = os.getenv("MODEL_PROVIDER", "openai")

    original_name = os.environ.get("MODEL_NAME")
    original_provider = os.environ.get("MODEL_PROVIDER")

    os.environ["MODEL_NAME"] = model_name
    os.environ["MODEL_PROVIDER"] = provider

    try:
        client = LLMClient()
    finally:
        if original_name is None:
            os.environ.pop("MODEL_NAME", None)
        else:
            os.environ["MODEL_NAME"] = original_name

        if original_provider is None:
            os.environ.pop("MODEL_PROVIDER", None)
        else:
            os.environ["MODEL_PROVIDER"] = original_provider

    logger.debug("llm_factory.client_created", model_key=model_key, model_name=model_name)
    return client
