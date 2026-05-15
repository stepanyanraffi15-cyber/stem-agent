from __future__ import annotations

import os
import threading

import structlog

from src.stem.llm_client import LLMClient

logger = structlog.get_logger(__name__)

_GROQ_DEFAULT_CHEAP: str = "llama-3.1-8b-instant"
_GROQ_DEFAULT_STRONG: str = "llama-3.3-70b-versatile"

_VALID_KEYS: frozenset[str] = frozenset(
    ("agent_review", "gradient", "variant_gen", "ewc_merge", "stem_phase")
)

_MAX_TOKENS_BY_ROLE: dict[str, int] = {
    "agent_review": 1024,
    "gradient":     2048,
    "variant_gen":  2048,
    "ewc_merge":    2048,
    "stem_phase":   4096,
}

_FACTORY_LOCK: threading.Lock = threading.Lock()


def get_max_tokens(model_key: str) -> int:
    """Return the max_tokens budget for the given model role."""
    if model_key not in _VALID_KEYS:
        raise ValueError(
            f"Unknown model_key {model_key!r}. Valid keys: {sorted(_VALID_KEYS)}"
        )
    return _MAX_TOKENS_BY_ROLE[model_key]


def _groq_model_routing() -> dict[str, str]:
    """Resolve Groq model ids from env (per-role, then cheap/strong, then defaults)."""
    return {
        "agent_review": os.getenv(
            "GROQ_MODEL_AGENT_REVIEW",
            os.getenv("MODEL_NAME_CHEAP", _GROQ_DEFAULT_CHEAP),
        ),
        "gradient": os.getenv(
            "GROQ_MODEL_GRADIENT",
            os.getenv("MODEL_NAME_STRONG", _GROQ_DEFAULT_STRONG),
        ),
        "variant_gen": os.getenv(
            "GROQ_MODEL_VARIANT_GEN",
            os.getenv("MODEL_NAME_STRONG", _GROQ_DEFAULT_STRONG),
        ),
        "ewc_merge": os.getenv(
            "GROQ_MODEL_EWC_MERGE",
            os.getenv("MODEL_NAME_STRONG", _GROQ_DEFAULT_STRONG),
        ),
        "stem_phase": os.getenv(
            "GROQ_MODEL_STEM_PHASE",
            os.getenv("MODEL_NAME_STRONG", _GROQ_DEFAULT_STRONG),
        ),
    }


def _openai_model_routing() -> dict[str, str]:
    """Resolve OpenAI model ids from MODEL_NAME_CHEAP / MODEL_NAME_STRONG."""
    return {
        "agent_review": os.getenv("MODEL_NAME_CHEAP", "gpt-4o-mini"),
        "gradient": os.getenv("MODEL_NAME_STRONG", "gpt-4o"),
        "variant_gen": os.getenv("MODEL_NAME_STRONG", "gpt-4o"),
        "ewc_merge": os.getenv("MODEL_NAME_STRONG", "gpt-4o"),
        "stem_phase": os.getenv("MODEL_NAME_STRONG", "gpt-4o"),
    }


def model_routing() -> dict[str, str]:
    """Active routing for the current process environment."""
    use_groq = os.getenv("GROQ_API_KEY") or os.getenv("MODEL_PROVIDER", "").lower() == "groq"
    return _groq_model_routing() if use_groq else _openai_model_routing()


def get_llm_client(model_key: str) -> LLMClient:
    """Return a configured LLMClient for the given model role.

    Thread-safe: the env mutation that LLMClient.__init__ reads is protected
    by _FACTORY_LOCK so concurrent callers cannot clobber each other's values.
    """
    if model_key not in _VALID_KEYS:
        raise ValueError(
            f"Unknown model_key {model_key!r}. Valid keys: {sorted(_VALID_KEYS)}"
        )

    model_name = model_routing()[model_key]
    provider = os.getenv("MODEL_PROVIDER", "openai")

    with _FACTORY_LOCK:
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
