"""Collector for Anthropic Claude models with summarized thinking (Regime 2: summarized_artifact)."""

import logging
import os
from typing import Any

import anthropic

from .base_collector import (
    ArtifactType,
    BaseCollector,
    FailureMetadata,
    Provider,
    TransparencyRegime,
)

logger = logging.getLogger(__name__)


class AnthropicCollector(BaseCollector):
    """Collects responses from Claude models with extended thinking.

    Claude Sonnet 4.6 returns summarized thinking blocks by default.
    The thinking blocks are lossy compressions of the internal reasoning.
    """

    provider = Provider.ANTHROPIC
    transparency_regime = TransparencyRegime.SUMMARIZED_ARTIFACT
    artifact_type = ArtifactType.THOUGHT_SUMMARY

    def __init__(
        self,
        model_name: str = "claude-sonnet-4-6-20260401",
        model_version: str = "",
        temperature: float = 1.0,  # Anthropic requires temp=1 with extended thinking
        max_tokens: int = 16384,
        thinking_budget: int | None = 10000,
        api_key: str | None = None,
    ):
        # Anthropic extended thinking requires temperature=1
        super().__init__(
            model_name=model_name,
            model_version=model_version,
            temperature=1.0,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
        )
        self._client = anthropic.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY")
        )

    def _call_api(self, prompt: str) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.model_name,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }

        # Enable extended thinking
        if self.thinking_budget:
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.thinking_budget,
            }

        response = self._client.messages.create(**kwargs)

        if hasattr(response, "model_dump"):
            return response.model_dump()
        return dict(response)

    def _extract_reasoning_artifact(self, raw_response: dict[str, Any]) -> str:
        content = raw_response.get("content", [])
        thinking_parts = []
        for block in content:
            if block.get("type") == "thinking":
                thinking_parts.append(block.get("thinking", ""))
            elif block.get("type") == "thinking_summary":
                thinking_parts.append(block.get("summary", ""))
        return "\n".join(thinking_parts)

    def _extract_final_answer(self, raw_response: dict[str, Any]) -> str:
        content = raw_response.get("content", [])
        text_parts = []
        for block in content:
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
        return "\n".join(text_parts)

    def _extract_content_blocks(self, raw_response: dict[str, Any]) -> list[dict[str, Any]]:
        return raw_response.get("content", [])

    def _extract_token_usage(self, raw_response: dict[str, Any]) -> dict[str, int | None]:
        usage = raw_response.get("usage", {})
        return {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "reasoning_tokens": None,
            "thinking_tokens": usage.get("thinking_tokens"),
        }

    def _check_failures(self, raw_response: dict[str, Any]) -> FailureMetadata:
        stop_reason = raw_response.get("stop_reason", "")
        content = raw_response.get("content", [])

        # Check for refusal
        final_text = self._extract_final_answer(raw_response).lower()
        refusal = any(
            phrase in final_text
            for phrase in ["i cannot", "i'm unable", "i must decline"]
        )

        return FailureMetadata(
            truncated=stop_reason == "max_tokens",
            refusal=refusal,
            incomplete=stop_reason not in ("end_turn", "stop_sequence", "tool_use"),
        )
