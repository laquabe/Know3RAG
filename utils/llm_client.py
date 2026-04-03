"""
Unified LLM client interface merging code/api.py (async OpenAI/Qwen)
and code/LLM_calls.py (local HuggingFace models).
"""
from __future__ import annotations
import asyncio
from typing import Dict, List, Optional

from config import LLMConfig


# ---------------------------------------------------------------------------
# Base interface
# ---------------------------------------------------------------------------

class BaseLLMClient:
    """Abstract base — every backend implements call() and call_batch()."""

    def call(self, messages: List[Dict[str, str]], max_tokens: int = 1024) -> str:
        raise NotImplementedError

    def call_batch(
        self, batch: List[List[Dict[str, str]]], max_tokens: int = 1024
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
        self.batch_size = config.api_batch_size
        self._client = AsyncOpenAI(api_key=config.openai_api_key)

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
            print(f"OpenAI request error: {e}")
            return ''

    async def _run_batch(self, batch: List[List[Dict]], max_tokens: int) -> List[str]:
        tasks = [self._fetch(msgs, max_tokens) for msgs in batch]
        return await asyncio.gather(*tasks)

    def call(self, messages: List[Dict], max_tokens: int = 1024) -> str:
        return asyncio.run(self._fetch(messages, max_tokens))

    def call_batch(self, batch: List[List[Dict]], max_tokens: int = 1024) -> List[str]:
        results = []
        for i in range(0, len(batch), self.batch_size):
            chunk = batch[i: i + self.batch_size]
            results.extend(asyncio.run(self._run_batch(chunk, max_tokens)))
        return results


class QwenAPIClient(BaseLLMClient):
    """Wraps the Qwen API via Alibaba DashScope (OpenAI-compatible interface)."""

    def __init__(self, config: LLMConfig):
        from openai import AsyncOpenAI
        self.model = config.qwen_api_model
        self.max_tokens = config.api_max_tokens
        self.timeout = config.api_timeout
        self.batch_size = config.api_batch_size
        self._client = AsyncOpenAI(
            api_key=config.qwen_api_key,
            base_url=config.qwen_base_url,
        )

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
            print(f"Qwen API request error: {e}")
            return ''

    async def _run_batch(self, batch: List[List[Dict]], max_tokens: int) -> List[str]:
        tasks = [self._fetch(msgs, max_tokens) for msgs in batch]
        return await asyncio.gather(*tasks)

    def call(self, messages: List[Dict], max_tokens: int = 1024) -> str:
        return asyncio.run(self._fetch(messages, max_tokens))

    def call_batch(self, batch: List[List[Dict]], max_tokens: int = 1024) -> List[str]:
        results = []
        for i in range(0, len(batch), self.batch_size):
            chunk = batch[i: i + self.batch_size]
            results.extend(asyncio.run(self._run_batch(chunk, max_tokens)))
        return results


# ---------------------------------------------------------------------------
# Local HuggingFace client
# ---------------------------------------------------------------------------

class LocalLLMClient(BaseLLMClient):
    """
    Wraps load_llm() + llm_call() from code/LLM_calls.py.
    Supported model names: Mistral | Llama | GLM3 | GLM4 | Baichuan | Yi | Qwen | Zephyr | Qwen_api
    """

    def __init__(self, config: LLMConfig):
        # Import lazily so the module can be imported without GPU
        from LLM_calls import load_llm  # type: ignore
        self.model_name = config.local_model_name
        self.max_new_tokens = config.local_max_new_tokens
        self.do_sample = config.local_do_sample

        loaded = load_llm(config.local_model_name, config.local_model_path)

        # load_llm returns either (model, tokenizer) or a pipeline
        if isinstance(loaded, tuple):
            self._model, self._tokenizer = loaded
            self._pipeline = None
        else:
            self._pipeline = loaded
            self._model, self._tokenizer = None, None

    def call(self, messages: List[Dict], max_tokens: int = 1024) -> str:
        from LLM_calls import llm_call  # type: ignore
        return llm_call(
            messages,
            self.model_name,
            model=self._model,
            tokenizer=self._tokenizer,
            pipeline=self._pipeline,
            do_sample=self.do_sample,
            max_new_tokens=max_tokens,
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
