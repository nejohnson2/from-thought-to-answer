"""Base collector interface and shared data schema for the CoT uncertainty study."""

import json
import logging
import time
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class Provider(str, Enum):
    OLLAMA = "ollama"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    OPENAI = "openai"


class TransparencyRegime(str, Enum):
    RAW_TRACE = "raw_trace"
    SUMMARIZED_ARTIFACT = "summarized_artifact"


class ArtifactType(str, Enum):
    RAW_TRACE = "raw_trace"
    THOUGHT_SUMMARY = "thought_summary"
    REASONING_SUMMARY = "reasoning_summary"
    NONE = "none"


class TaskBucket(str, Enum):
    EASY_REASONING = "easy_reasoning"
    HARD_REASONING = "hard_reasoning"
    UNANSWERABLE = "unanswerable"
    UNDERSPECIFIED = "underspecified"
    FACTUAL_QA = "factual_qa"


# ---------------------------------------------------------------------------
# Schema models
# ---------------------------------------------------------------------------

class QuestionMetadata(BaseModel):
    source_dataset: str
    ground_truth: str | None = None
    is_answerable: bool = True
    difficulty: str | None = None


class ParsedAnswer(BaseModel):
    final_answer: str = ""
    abstain: bool = False
    confidence_0_100: int = -1
    justification: str = ""


class GenerationMetadata(BaseModel):
    temperature: float = 0.0
    max_tokens: int = 0
    thinking_budget: int | None = None
    latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int | None = None
    thinking_tokens: int | None = None
    timestamp: str = ""
    sample_index: int = 0


class FailureMetadata(BaseModel):
    truncated: bool = False
    refusal: bool = False
    parse_failure: bool = False
    incomplete: bool = False
    error_message: str | None = None


class ResponseRecord(BaseModel):
    """Single model response — the canonical data record for this study."""

    prompt_id: str
    model: str
    provider: Provider
    model_version: str = ""
    transparency_regime: TransparencyRegime
    task_bucket: TaskBucket
    interpolated_prompt: str
    question_text: str
    question_metadata: QuestionMetadata

    reasoning_artifact: str = ""
    artifact_type: ArtifactType = ArtifactType.NONE
    content_blocks: list[dict[str, Any]] = Field(default_factory=list)
    final_answer_raw: str = ""

    parsed_answer: ParsedAnswer = Field(default_factory=ParsedAnswer)
    generation_metadata: GenerationMetadata = Field(default_factory=GenerationMetadata)
    failure_metadata: FailureMetadata = Field(default_factory=FailureMetadata)

    raw_provider_payload: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Task item (input)
# ---------------------------------------------------------------------------

class TaskItem(BaseModel):
    """A single prompt from the task suite."""

    prompt_id: str
    question_text: str
    task_bucket: TaskBucket
    metadata: QuestionMetadata


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = """\
Reason carefully about the following question, then answer.

Question: {question}

Respond in the following JSON format:
{{
  "final_answer": "<your answer>",
  "abstain": <true or false>,
  "confidence": <0-100>,
  "justification": "<one sentence explaining your confidence>"
}}"""


def build_prompt(question: str) -> str:
    """Interpolate the question into the standardized prompt template."""
    return PROMPT_TEMPLATE.format(question=question)


# ---------------------------------------------------------------------------
# Answer parsing
# ---------------------------------------------------------------------------

def parse_answer_json(raw_text: str) -> tuple[ParsedAnswer, bool]:
    """Attempt to parse the structured answer from model output.

    Returns (ParsedAnswer, success). On failure, returns a default ParsedAnswer
    with whatever could be extracted, and success=False.
    """
    text = raw_text.strip()

    # Try to find JSON block in the output
    json_start = text.find("{")
    json_end = text.rfind("}") + 1

    if json_start == -1 or json_end == 0:
        return ParsedAnswer(final_answer=text), False

    json_str = text[json_start:json_end]

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return ParsedAnswer(final_answer=text), False

    try:
        parsed = ParsedAnswer(
            final_answer=str(data.get("final_answer", "")),
            abstain=bool(data.get("abstain", False)),
            confidence_0_100=int(data.get("confidence", -1)),
            justification=str(data.get("justification", "")),
        )
        return parsed, True
    except (ValueError, TypeError):
        return ParsedAnswer(final_answer=text), False


# ---------------------------------------------------------------------------
# Storage helpers
# ---------------------------------------------------------------------------

def save_record(record: ResponseRecord, output_dir: Path) -> None:
    """Append a ResponseRecord to the appropriate JSONL file and save raw payload."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Determine filenames from model name
    safe_model = record.model.replace("/", "_").replace(":", "_")
    jsonl_path = output_dir / f"{safe_model}.jsonl"

    # Save normalized record
    with open(jsonl_path, "a") as f:
        f.write(record.model_dump_json() + "\n")

    # Save raw payload separately
    payload_dir = output_dir.parent / "raw_payloads" / safe_model
    payload_dir.mkdir(parents=True, exist_ok=True)
    payload_path = payload_dir / f"{record.prompt_id}_s{record.generation_metadata.sample_index}.json"
    with open(payload_path, "w") as f:
        json.dump(record.raw_provider_payload, f, indent=2, default=str)


def load_records(jsonl_path: Path) -> list[ResponseRecord]:
    """Load all ResponseRecords from a JSONL file."""
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(ResponseRecord.model_validate_json(line))
    return records


def load_tasks(tasks_dir: Path) -> list[TaskItem]:
    """Load all task items from the tasks directory."""
    tasks = []
    for jsonl_path in sorted(tasks_dir.glob("*.jsonl")):
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    tasks.append(TaskItem.model_validate_json(line))
    return tasks


# ---------------------------------------------------------------------------
# Base collector
# ---------------------------------------------------------------------------

class BaseCollector(ABC):
    """Abstract base for provider-specific collectors."""

    provider: Provider
    transparency_regime: TransparencyRegime
    artifact_type: ArtifactType
    model_name: str
    model_version: str

    def __init__(
        self,
        model_name: str,
        model_version: str = "",
        temperature: float = 0.0,
        max_tokens: int = 16384,
        thinking_budget: int | None = None,
    ):
        self.model_name = model_name
        self.model_version = model_version
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.thinking_budget = thinking_budget

    @abstractmethod
    def _call_api(self, prompt: str) -> dict[str, Any]:
        """Make the raw API call. Returns the raw provider response as a dict."""
        ...

    @abstractmethod
    def _extract_reasoning_artifact(self, raw_response: dict[str, Any]) -> str:
        """Extract the reasoning artifact (trace or summary) from the response."""
        ...

    @abstractmethod
    def _extract_final_answer(self, raw_response: dict[str, Any]) -> str:
        """Extract the final answer text from the response."""
        ...

    @abstractmethod
    def _extract_content_blocks(self, raw_response: dict[str, Any]) -> list[dict[str, Any]]:
        """Extract the full content-block array from the response."""
        ...

    @abstractmethod
    def _extract_token_usage(self, raw_response: dict[str, Any]) -> dict[str, int | None]:
        """Extract token usage info. Returns dict with keys:
        input_tokens, output_tokens, reasoning_tokens, thinking_tokens."""
        ...

    @abstractmethod
    def _check_failures(self, raw_response: dict[str, Any]) -> FailureMetadata:
        """Check for truncation, refusal, parse failure, etc."""
        ...

    def collect(self, task: TaskItem, sample_index: int = 0) -> ResponseRecord:
        """Collect a single response for a task item."""
        prompt = build_prompt(task.question_text)

        start_time = time.monotonic()
        try:
            raw_response = self._call_api(prompt)
        except Exception as e:
            logger.error("API call failed for %s on %s: %s", self.model_name, task.prompt_id, e)
            return ResponseRecord(
                prompt_id=task.prompt_id,
                model=self.model_name,
                provider=self.provider,
                model_version=self.model_version,
                transparency_regime=self.transparency_regime,
                task_bucket=task.task_bucket,
                interpolated_prompt=prompt,
                question_text=task.question_text,
                question_metadata=task.metadata,
                failure_metadata=FailureMetadata(
                    incomplete=True,
                    error_message=str(e),
                ),
                generation_metadata=GenerationMetadata(
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    thinking_budget=self.thinking_budget,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    sample_index=sample_index,
                ),
            )
        latency_ms = int((time.monotonic() - start_time) * 1000)

        reasoning_artifact = self._extract_reasoning_artifact(raw_response)
        final_answer_raw = self._extract_final_answer(raw_response)
        content_blocks = self._extract_content_blocks(raw_response)
        token_usage = self._extract_token_usage(raw_response)
        failure_metadata = self._check_failures(raw_response)

        parsed_answer, parse_ok = parse_answer_json(final_answer_raw)
        if not parse_ok:
            failure_metadata.parse_failure = True

        record = ResponseRecord(
            prompt_id=task.prompt_id,
            model=self.model_name,
            provider=self.provider,
            model_version=self.model_version,
            transparency_regime=self.transparency_regime,
            task_bucket=task.task_bucket,
            interpolated_prompt=prompt,
            question_text=task.question_text,
            question_metadata=task.metadata,
            reasoning_artifact=reasoning_artifact,
            artifact_type=self.artifact_type,
            content_blocks=content_blocks,
            final_answer_raw=final_answer_raw,
            parsed_answer=parsed_answer,
            generation_metadata=GenerationMetadata(
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                thinking_budget=self.thinking_budget,
                latency_ms=latency_ms,
                input_tokens=token_usage.get("input_tokens", 0),
                output_tokens=token_usage.get("output_tokens", 0),
                reasoning_tokens=token_usage.get("reasoning_tokens"),
                thinking_tokens=token_usage.get("thinking_tokens"),
                timestamp=datetime.now(timezone.utc).isoformat(),
                sample_index=sample_index,
            ),
            failure_metadata=failure_metadata,
            raw_provider_payload=raw_response,
        )

        return record
