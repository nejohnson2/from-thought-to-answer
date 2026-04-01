"""Collector for Google Gemini models with thought summaries (Regime 2: summarized_artifact)."""

import logging
import os
from typing import Any

from google import genai
from google.genai import types

from .base_collector import (
    ArtifactType,
    BaseCollector,
    FailureMetadata,
    Provider,
    TransparencyRegime,
)

logger = logging.getLogger(__name__)


class GoogleCollector(BaseCollector):
    """Collects responses from Gemini thinking models.

    Gemini 2.5 Flash returns thought summaries (not raw thoughts) when
    thinkingConfig is enabled. Google docs are explicit that these are
    summarized versions of the model's reasoning.
    """

    provider = Provider.GOOGLE
    transparency_regime = TransparencyRegime.SUMMARIZED_ARTIFACT
    artifact_type = ArtifactType.THOUGHT_SUMMARY

    def __init__(
        self,
        model_name: str = "gemini-2.5-flash",
        model_version: str = "",
        temperature: float = 0.0,
        max_tokens: int = 16384,
        thinking_budget: int | None = 10000,
        api_key: str | None = None,
    ):
        super().__init__(
            model_name=model_name,
            model_version=model_version,
            temperature=temperature,
            max_tokens=max_tokens,
            thinking_budget=thinking_budget,
        )
        self._client = genai.Client(
            api_key=api_key or os.environ.get("GOOGLE_API_KEY")
        )

    def _call_api(self, prompt: str) -> dict[str, Any]:
        config = types.GenerateContentConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens,
            thinking_config=types.ThinkingConfig(
                thinking_budget=self.thinking_budget,
                include_thoughts=True,
            ),
            response_mime_type="application/json",
        )

        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=config,
        )

        return self._response_to_dict(response)

    def _response_to_dict(self, response: Any) -> dict[str, Any]:
        """Convert Gemini response to a serializable dict."""
        result: dict[str, Any] = {}

        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]
            parts = []
            if hasattr(candidate, "content") and candidate.content:
                for part in candidate.content.parts:
                    part_dict: dict[str, Any] = {}
                    if hasattr(part, "thought") and part.thought:
                        part_dict["type"] = "thought"
                        part_dict["thought"] = True
                        part_dict["text"] = part.text or ""
                    else:
                        part_dict["type"] = "text"
                        part_dict["thought"] = False
                        part_dict["text"] = part.text or ""
                    parts.append(part_dict)
            result["parts"] = parts
            result["finish_reason"] = str(getattr(candidate, "finish_reason", ""))
        else:
            result["parts"] = []
            result["finish_reason"] = "unknown"

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            usage = response.usage_metadata
            result["usage"] = {
                "prompt_token_count": getattr(usage, "prompt_token_count", 0),
                "candidates_token_count": getattr(usage, "candidates_token_count", 0),
                "total_token_count": getattr(usage, "total_token_count", 0),
                "thinking_token_count": getattr(usage, "thinking_token_count", 0),
            }
        else:
            result["usage"] = {}

        return result

    def _extract_reasoning_artifact(self, raw_response: dict[str, Any]) -> str:
        parts = raw_response.get("parts", [])
        thought_parts = [p["text"] for p in parts if p.get("thought")]
        return "\n".join(thought_parts)

    def _extract_final_answer(self, raw_response: dict[str, Any]) -> str:
        parts = raw_response.get("parts", [])
        text_parts = [p["text"] for p in parts if not p.get("thought")]
        return "\n".join(text_parts)

    def _extract_content_blocks(self, raw_response: dict[str, Any]) -> list[dict[str, Any]]:
        return raw_response.get("parts", [])

    def _extract_token_usage(self, raw_response: dict[str, Any]) -> dict[str, int | None]:
        usage = raw_response.get("usage", {})
        return {
            "input_tokens": usage.get("prompt_token_count", 0),
            "output_tokens": usage.get("candidates_token_count", 0),
            "reasoning_tokens": None,
            "thinking_tokens": usage.get("thinking_token_count"),
        }

    def _check_failures(self, raw_response: dict[str, Any]) -> FailureMetadata:
        finish_reason = raw_response.get("finish_reason", "")
        final_text = self._extract_final_answer(raw_response).lower()

        return FailureMetadata(
            truncated="MAX_TOKENS" in finish_reason.upper(),
            refusal="SAFETY" in finish_reason.upper() or "RECITATION" in finish_reason.upper(),
            incomplete=finish_reason not in ("STOP", "FinishReason.STOP", ""),
        )
