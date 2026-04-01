"""Collector for OpenAI reasoning models with reasoning summaries (Regime 2: summarized_artifact)."""

import logging
import os
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


class OpenAICollector(BaseCollector):
    """Collects responses from OpenAI o-series reasoning models.

    OpenAI reasoning models (o4-mini, o3) do not expose raw reasoning tokens.
    When opted in, they return reasoning summaries via the reasoning output
    item's summary array. Raw reasoning tokens are billed but invisible.
    """

    provider = Provider.OPENAI
    transparency_regime = TransparencyRegime.SUMMARIZED_ARTIFACT
    artifact_type = ArtifactType.REASONING_SUMMARY

    def __init__(
        self,
        model_name: str = "o4-mini",
        model_version: str = "",
        temperature: float = 1.0,  # o-series models have limited temp control
        max_tokens: int = 16384,
        thinking_budget: int | None = None,
        api_key: str | None = None,
        reasoning_effort: str = "medium",
    ):
        super().__init__(
            model_name=model_name,
            model_version=model_version,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
        )
        self.reasoning_effort = reasoning_effort
        self._client = openai.OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY")
        )

    def _call_api(self, prompt: str) -> dict[str, Any]:
        # Use the Responses API for o-series models
        response = self._client.responses.create(
            model=self.model_name,
            input=[{"role": "user", "content": prompt}],
            reasoning={"effort": self.reasoning_effort, "summary": "detailed"},
            max_output_tokens=self.max_tokens,
        )

        return self._response_to_dict(response)

    def _response_to_dict(self, response: Any) -> dict[str, Any]:
        """Convert OpenAI response to a serializable dict."""
        result: dict[str, Any] = {
            "id": getattr(response, "id", ""),
            "model": getattr(response, "model", ""),
            "output": [],
            "usage": {},
        }

        # Extract output items
        if hasattr(response, "output"):
            for item in response.output:
                item_dict: dict[str, Any] = {
                    "type": getattr(item, "type", ""),
                    "id": getattr(item, "id", ""),
                }

                if item_dict["type"] == "reasoning":
                    # Extract reasoning summary
                    summary_parts = []
                    if hasattr(item, "summary") and item.summary:
                        for summary_item in item.summary:
                            summary_parts.append({
                                "type": getattr(summary_item, "type", ""),
                                "text": getattr(summary_item, "text", ""),
                            })
                    item_dict["summary"] = summary_parts

                elif item_dict["type"] == "message":
                    # Extract message content
                    content_parts = []
                    if hasattr(item, "content") and item.content:
                        for content_item in item.content:
                            content_parts.append({
                                "type": getattr(content_item, "type", ""),
                                "text": getattr(content_item, "text", ""),
                            })
                    item_dict["content"] = content_parts

                result["output"].append(item_dict)

        # Extract usage
        if hasattr(response, "usage") and response.usage:
            usage = response.usage
            result["usage"] = {
                "input_tokens": getattr(usage, "input_tokens", 0),
                "output_tokens": getattr(usage, "output_tokens", 0),
            }
            # Extract reasoning token details
            if hasattr(usage, "output_tokens_details") and usage.output_tokens_details:
                details = usage.output_tokens_details
                result["usage"]["reasoning_tokens"] = getattr(
                    details, "reasoning_tokens", None
                )

        return result

    def _extract_reasoning_artifact(self, raw_response: dict[str, Any]) -> str:
        output = raw_response.get("output", [])
        summary_texts = []
        for item in output:
            if item.get("type") == "reasoning":
                for summary_part in item.get("summary", []):
                    text = summary_part.get("text", "")
                    if text:
                        summary_texts.append(text)
        return "\n".join(summary_texts)

    def _extract_final_answer(self, raw_response: dict[str, Any]) -> str:
        output = raw_response.get("output", [])
        text_parts = []
        for item in output:
            if item.get("type") == "message":
                for content_part in item.get("content", []):
                    text = content_part.get("text", "")
                    if text:
                        text_parts.append(text)
        return "\n".join(text_parts)

    def _extract_content_blocks(self, raw_response: dict[str, Any]) -> list[dict[str, Any]]:
        return raw_response.get("output", [])

    def _extract_token_usage(self, raw_response: dict[str, Any]) -> dict[str, int | None]:
        usage = raw_response.get("usage", {})
        return {
            "input_tokens": usage.get("input_tokens", 0),
            "output_tokens": usage.get("output_tokens", 0),
            "reasoning_tokens": usage.get("reasoning_tokens"),
            "thinking_tokens": None,
        }

    def _check_failures(self, raw_response: dict[str, Any]) -> FailureMetadata:
        # Check if we got any message output at all
        output = raw_response.get("output", [])
        has_message = any(item.get("type") == "message" for item in output)
        final_text = self._extract_final_answer(raw_response).lower()

        return FailureMetadata(
            truncated=not has_message and len(output) > 0,
            refusal=any(
                phrase in final_text
                for phrase in ["i cannot", "i'm unable", "i must decline"]
            ),
            incomplete=not has_message,
        )
