"""Central OpenAI-SDK call layer: cached clients + jittered retry/backoff.

Every generation/QA/embedding call in the pipeline routes through `call_chat` /
`call_embeddings` so retry, client reuse, and cost tracking live in one place.
All providers are reached via the OpenAI SDK
through a per-model `base_url`.
"""

import logging
from functools import lru_cache

import openai
from openai import APIConnectionError, APITimeoutError, InternalServerError, RateLimitError
from tenacity import (
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random_exponential,
)

from mistake_honesty_eval.utils import CostTracker, EmptyCompletionError

logger = logging.getLogger(__name__)

# Only transient failures are retried; auth/bad-request errors propagate immediately.
# EmptyCompletionError is in here because a provider can drop a stream mid-response and
# still return HTTP 200 (see `call_chat`), which no SDK-level exception covers.
RETRYABLE_EXCEPTIONS = (
    RateLimitError,
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    EmptyCompletionError,
)

MAX_ATTEMPTS = 6

# Jittered exponential backoff so pool workers don't retry in lockstep after a shared 429.
RETRY_KWARGS = dict(
    retry=retry_if_exception_type(RETRYABLE_EXCEPTIONS),
    wait=wait_random_exponential(multiplier=1, max=60),
    stop=stop_after_attempt(MAX_ATTEMPTS),
    before_sleep=before_sleep_log(logger, logging.WARNING),
    reraise=True,
)


@lru_cache(maxsize=None)
def _client(api_key: str, base_url: str | None) -> openai.OpenAI:
    """Return a cached OpenAI client for the given credentials.

    The OpenAI client is thread-safe, so one instance is shared across worker
    threads. `max_retries=0` disables the SDK's own retries so they don't stack
    multiplicatively on top of the tenacity retry below.
    """
    return openai.OpenAI(api_key=api_key, base_url=base_url, max_retries=0)


@retry(**RETRY_KWARGS)
def call_chat(
    params: dict,
    messages: list[dict],
    *,
    extra_body: dict | None = None,
    response_format: dict | None = None,
    temperature: float | None = None,
    cost_tracker: CostTracker | None = None,
) -> str:
    """Call chat.completions with retry/backoff; return the message content string.

    Records token usage/cost on `cost_tracker` when provided (thread-safe).

    A response with no content raises `EmptyCompletionError`, which is retryable. This
    is not hypothetical: OpenRouter returns HTTP 200 with `content: null` and
    `finish_reason: "error"` when an upstream provider drops the stream mid-response, so
    no SDK exception is raised and nothing else would catch it. Without this guard the
    empty string flows on as if it were the model's answer -- silently accepted as an
    assistant turn by `generate_erroneous_trajectory`, or surfaced by the tag extractors as a
    misleading "refusal or format error".
    """
    client = _client(params["api_key"], params.get("base_url"))
    kwargs: dict = {"model": params["model"], "messages": messages}
    if extra_body is not None:
        kwargs["extra_body"] = extra_body
    if response_format is not None:
        kwargs["response_format"] = response_format
    if temperature is not None:
        kwargs["temperature"] = temperature
    response = client.chat.completions.create(**kwargs)
    if cost_tracker is not None:
        cost_tracker.add_chat_usage(params, response.usage)
    choice = response.choices[0]
    content = choice.message.content
    if not (content or "").strip():
        raise EmptyCompletionError(
            f"{params['model']} returned no content (finish_reason={choice.finish_reason!r})"
        )
    return content


@retry(**RETRY_KWARGS)
def call_embeddings(params: dict, inputs: list[str], *, cost_tracker: CostTracker | None = None) -> list:
    """Call embeddings.create with retry/backoff; return `response.data` (order matches `inputs`)."""
    client = _client(params["api_key"], params.get("base_url"))
    response = client.embeddings.create(model=params["model"], input=inputs)
    if cost_tracker is not None:
        cost_tracker.add_embedding_usage(params, response.usage)
    return response.data
