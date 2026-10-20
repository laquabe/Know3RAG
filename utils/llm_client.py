"""
Unified LLM client interface merging code/api.py (async OpenAI/Qwen)
and refactored local HuggingFace model utilities.
"""
from __future__ import annotations
import asyncio
import os
from typing import Dict, List, Optional

from config import LLMConfig


class LLMCallError(RuntimeError):
    """A request failed; never represent transport errors as empty answers."""


class LLMBatchError(LLMCallError):
    """Ordered partial results: successful strings or per-request exceptions."""

    def __init__(self, results):
        self.results = results
        super().__init__(f"{sum(isinstance(r, Exception) for r in results)} LLM request(s) failed")


def _token_limit(value, default):
    value = default if value is None else value
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError("max_tokens must be a positive integer")
    return value


def _batch_size(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


def validate_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ValueError("Expected a non-empty list of {role, content} messages")
    for message in messages:
        if (not isinstance(message, dict)
                or message.get("role") not in ("system", "user", "assistant")
                or not isinstance(message.get("content"), str)):
            raise ValueError("Each message must contain a supported role and string content")


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class BaseLLMClient:
    """Abstract base — every backend implements call() and call_batch()."""

    def call(self, messages: List[Dict[str, str]], max_tokens: Optional[int] = None) -> str:
        raise NotImplementedError

    def call_batch(
        self, batch: List[List[Dict[str, str]]], max_tokens: Optional[int] = None
    ) -> List[str]:
        """Default: sequential loop. Override for true parallelism."""
        return [self.call(msgs, max_tokens) for msgs in batch]


# ---------------------------------------------------------------------------
# API-backed clients (OpenAI / Qwen-via-DashScope)
# ---------------------------------------------------------------------------

class OpenAIClient(BaseLLMClient):
    """Wraps the async OpenAI Chat Completions API (gpt-* models)."""

    def __init__(self, config: LLMConfig):
        from openai import AsyncOpenAI
        self.model = config.openai_model
        self.max_tokens = config.api_max_tokens
        self.timeout = config.api_timeout
        self.batch_size = _batch_size(config.api_batch_size, "api_batch_size")
        self._client = AsyncOpenAI(api_key=config.openai_api_key or os.environ.get("OPENAI_API_KEY"))

    async def _fetch(self, messages: List[Dict], max_tokens: int) -> str:
        try:
            response = await asyncio.wait_for(
                self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=max_tokens,
                ),
                timeout=self.timeout,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            raise LLMCallError(f"{self.model}: {type(e).__name__}: {e}") from e

    async def _run_batch(self, batch: List[List[Dict]], max_tokens: int) -> List[str]:
        tasks = [self._fetch(msgs, max_tokens) for msgs in batch]
        return await asyncio.gather(*tasks, return_exceptions=True)

    def call(self, messages: List[Dict], max_tokens: Optional[int] = None) -> str:
        validate_messages(messages)
        return asyncio.run(self._fetch(messages, _token_limit(max_tokens, self.max_tokens)))

    async def _all_batches(self, batch, max_tokens):
        results = []
        for i in range(0, len(batch), self.batch_size):
            chunk = batch[i: i + self.batch_size]
            results.extend(await self._run_batch(chunk, max_tokens))
        return results

    def call_batch(self, batch: List[List[Dict]], max_tokens: Optional[int] = None) -> List[str]:
        for messages in batch:
            validate_messages(messages)
        if not batch:
            return []
        results = asyncio.run(self._all_batches(batch, _token_limit(max_tokens, self.max_tokens)))
        if any(isinstance(result, Exception) for result in results):
            raise LLMBatchError(results)
        return results


class QwenAPIClient(OpenAIClient):
    """Wraps the Qwen API via Alibaba DashScope (OpenAI-compatible interface)."""

    def __init__(self, config: LLMConfig):
        from openai import AsyncOpenAI
        self.model = config.qwen_api_model
        self.max_tokens = config.api_max_tokens
        self.timeout = config.api_timeout
        self.batch_size = _batch_size(config.api_batch_size, "api_batch_size")
        self._client = AsyncOpenAI(
            api_key=config.qwen_api_key or os.environ.get("DASHSCOPE_API_KEY"),
            base_url=config.qwen_base_url,
        )


# ---------------------------------------------------------------------------
# Local HuggingFace client
# ---------------------------------------------------------------------------

class LocalLLMClient(BaseLLMClient):
    """
    Wraps local HuggingFace model loading and inference.
    Supported model names: Mistral | Llama | GLM3 | GLM4 | Baichuan | Yi | Qwen | Zephyr
    """

    def __init__(self, config: LLMConfig):
        # Import lazily so the module can be imported without GPU
        from utils.local_llm import load_local_llm
        self.model_name = config.local_model_name
        self.max_new_tokens = config.local_max_new_tokens
        self.do_sample = config.local_do_sample
        self.batch_size = _batch_size(config.local_batch_size, "local_batch_size")

        loaded = load_local_llm(config.local_model_name, config.local_model_path)

        # load_llm returns either (model, tokenizer) or a pipeline
        if isinstance(loaded, tuple):
            self._model, self._tokenizer = loaded
            self._pipeline = None
        else:
            self._pipeline = loaded
            self._model, self._tokenizer = None, None

    def call(self, messages: List[Dict], max_tokens: Optional[int] = None) -> str:
        from utils.local_llm import call_local_llm
        validate_messages(messages)
        return call_local_llm(
            messages,
            self.model_name,
            model=self._model,
            tokenizer=self._tokenizer,
            pipeline=self._pipeline,
            do_sample=self.do_sample,
            max_new_tokens=_token_limit(max_tokens, self.max_new_tokens),
        )

    def call_batch(self, batch: List[List[Dict]], max_tokens: Optional[int] = None) -> List[str]:
        from utils.local_llm import call_local_llm_batch
        for messages in batch:
            validate_messages(messages)
        self.last_batch_events = []
        return call_local_llm_batch(
            batch, self.model_name, model=self._model, tokenizer=self._tokenizer,
            pipeline=self._pipeline, do_sample=self.do_sample,
            max_new_tokens=_token_limit(max_tokens, self.max_new_tokens),
            batch_size=self.batch_size,
            batch_events=self.last_batch_events,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def create_llm_client(config: LLMConfig) -> BaseLLMClient:
    """
    Returns the appropriate LLM client based on config.local_model_name.
    'api'      -> OpenAIClient
    'qwen_api' -> QwenAPIClient
    others     -> LocalLLMClient (HuggingFace)
    """
    name = config.local_model_name.lower()
    if name == 'api':
        return OpenAIClient(config)
    elif name == 'qwen_api':
        return QwenAPIClient(config)
    else:
        return LocalLLMClient(config)
