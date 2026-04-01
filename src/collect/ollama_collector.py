"""Collector for Ollama models with full thinking traces (Regime 1: raw_trace)."""

import logging
from typing import Any

import ollama as ollama_client

from .base_collector import (
    ArtifactType,
    BaseCollector,
    FailureMetadata,
    Provider,
    TransparencyRegime,
)

logger = logging.getLogger(__name__)


class OllamaCollector(BaseCollector):
    """Collects responses from Ollama reasoning models.

    Ollama separates thinking from content via message.thinking when
    the `think` option is enabled. This is the only regime that returns
    full, uncompressed reasoning traces.
    """

    provider = Provider.OLLAMA
    transparency_regime = TransparencyRegime.RAW_TRACE
    artifact_type = ArtifactType.RAW_TRACE

    def __init__(
        self,
        model_name: str,
        model_version: str = "",
        temperature: float = 0.0,
        max_tokens: int = 16384,
        thinking_budget: int | None = None,
        host: str | None = None,
    ):
        super().__init__(
            model_name=model_name,
            model_version=model_version,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
        )
        self.host = host
        self._client = ollama_client.Client(host=host) if host else ollama_client.Client()

    def _call_api(self, prompt: str) -> dict[str, Any]:
        response = self._client.chat(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            options={
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
            think=True,
            format="json",
        )
        # ollama returns a ChatResponse object; convert to dict
        if hasattr(response, "model_dump"):
            return response.model_dump()
        return dict(response)

    def _extract_reasoning_artifact(self, raw_response: dict[str, Any]) -> str:
        message = raw_response.get("message", {})
        return message.get("thinking", "") or ""

    def _extract_final_answer(self, raw_response: dict[str, Any]) -> str:
        message = raw_response.get("message", {})
        return message.get("content", "") or ""

    def _extract_content_blocks(self, raw_response: dict[str, Any]) -> list[dict[str, Any]]:
        message = raw_response.get("message", {})
        blocks = []
        thinking = message.get("thinking", "")
        if thinking:
            blocks.append({"type": "thinking", "text": thinking})
        content = message.get("content", "")
        if content:
            blocks.append({"type": "content", "text": content})
        return blocks

    def _extract_token_usage(self, raw_response: dict[str, Any]) -> dict[str, int | None]:
        return {
            "input_tokens": raw_response.get("prompt_eval_count", 0),
            "output_tokens": raw_response.get("eval_count", 0),
            "reasoning_tokens": None,
            "thinking_tokens": None,
        }

    def _check_failures(self, raw_response: dict[str, Any]) -> FailureMetadata:
        done_reason = raw_response.get("done_reason", "")
        message = raw_response.get("message", {})
        content = message.get("content", "") or ""

        return FailureMetadata(
            truncated=done_reason == "length",
            refusal=any(
                phrase in content.lower()
                for phrase in ["i cannot", "i'm unable", "i refuse"]
            ),
            incomplete=not raw_response.get("done", True),
        )
