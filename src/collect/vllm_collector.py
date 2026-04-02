"""Collector for open-weight models served via vLLM (Regime 1: raw_trace).

vLLM serves an OpenAI-compatible API. Reasoning models like DeepSeek-R1 and
Qwen3 produce <think>...</think> blocks in their output, which we parse to
separate the reasoning trace from the final answer.
"""

import logging
import re
from typing import Any

import openai

from .base_collector import (
    ArtifactType,
    BaseCollector,
    FailureMetadata,
    Provider,
    TransparencyRegime,
)

logger = logging.getLogger(__name__)


def _split_thinking(text: str) -> tuple[str, str]:
    """Split <think>...</think> blocks from the final content.

    Returns (thinking_text, content_text).
    """
    # Match <think>...</think> blocks (may span multiple lines)
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    thinking_parts = think_pattern.findall(text)
    thinking = "\n".join(thinking_parts).strip()

    # Everything outside <think> blocks is the final content
    content = think_pattern.sub("", text).strip()

    return thinking, content


class VLLMCollector(BaseCollector):
    """Collects responses from models served via vLLM.

    vLLM exposes an OpenAI-compatible API on localhost. Reasoning models
    (DeepSeek-R1, Qwen3) emit <think>...</think> blocks that we parse
    to extract the full reasoning trace.
    """

    provider = Provider.OLLAMA  # Reuse OLLAMA provider enum for open-weight
    transparency_regime = TransparencyRegime.RAW_TRACE
    artifact_type = ArtifactType.RAW_TRACE

    def __init__(
        self,
        model_name: str,
        model_version: str = "",
        temperature: float = 0.0,
        max_tokens: int = 8192,
        thinking_budget: int | None = None,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "dummy",
    ):
        super().__init__(
            model_name=model_name,
            model_version=model_version,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
        )
        self.base_url = base_url
        self._client = openai.OpenAI(base_url=base_url, api_key=api_key)

        # Resolve the actual model name served by vLLM
        # vLLM uses the HF model path as the model name
        self._served_model = self._resolve_model_name()

    def _resolve_model_name(self) -> str:
        """Get the actual model name from the vLLM server."""
        try:
            models = self._client.models.list()
            if models.data:
                served = models.data[0].id
                logger.info("vLLM serving model: %s", served)
                return served
        except Exception as e:
            logger.warning("Could not query vLLM models endpoint: %s", e)
        return self.model_name

    def _call_api(self, prompt: str) -> dict[str, Any]:
        response = self._client.chat.completions.create(
            model=self._served_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )

        # Convert to dict
        return response.model_dump() if hasattr(response, "model_dump") else dict(response)

    def _extract_reasoning_artifact(self, raw_response: dict[str, Any]) -> str:
        full_text = self._get_full_text(raw_response)
        thinking, _ = _split_thinking(full_text)
        return thinking

    def _extract_final_answer(self, raw_response: dict[str, Any]) -> str:
        full_text = self._get_full_text(raw_response)
        _, content = _split_thinking(full_text)
        return content

    def _extract_content_blocks(self, raw_response: dict[str, Any]) -> list[dict[str, Any]]:
        full_text = self._get_full_text(raw_response)
        thinking, content = _split_thinking(full_text)
        blocks = []
        if thinking:
            blocks.append({"type": "thinking", "text": thinking})
        if content:
            blocks.append({"type": "content", "text": content})
        return blocks

    def _extract_token_usage(self, raw_response: dict[str, Any]) -> dict[str, int | None]:
        usage = raw_response.get("usage", {})
        return {
            "input_tokens": usage.get("prompt_tokens", 0),
            "output_tokens": usage.get("completion_tokens", 0),
            "reasoning_tokens": None,
            "thinking_tokens": None,
        }

    def _check_failures(self, raw_response: dict[str, Any]) -> FailureMetadata:
        choices = raw_response.get("choices", [])
        if not choices:
            return FailureMetadata(incomplete=True, error_message="No choices in response")

        choice = choices[0]
        finish_reason = choice.get("finish_reason", "")
        content = self._get_full_text(raw_response).lower()

        return FailureMetadata(
            truncated=finish_reason == "length",
            refusal=any(
                phrase in content
                for phrase in ["i cannot", "i'm unable", "i refuse"]
            ),
            incomplete=finish_reason not in ("stop", "length"),
        )

    def _get_full_text(self, raw_response: dict[str, Any]) -> str:
        """Extract the full text from the OpenAI-compatible response."""
        choices = raw_response.get("choices", [])
        if not choices:
            return ""
        message = choices[0].get("message", {})
        return message.get("content", "") or ""
