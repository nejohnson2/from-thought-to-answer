"""LLM judge for uncertainty classification (secondary annotation layer).

Uses Llama 4 via Ollama to classify uncertainty level, answer stance,
and transmission/suppression in model responses.
"""

import json
import logging
from typing import Any

import ollama as ollama_client

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Judge prompt
# ---------------------------------------------------------------------------

JUDGE_SYSTEM_PROMPT = """\
You are an expert annotator for a research study on uncertainty in LLM reasoning.
You will be shown a model's reasoning artifact (thinking trace or summary) and its
final answer to a question. Your job is to classify the uncertainty characteristics
of both the reasoning and the answer.

Respond ONLY in the following JSON format:
{
  "artifact_uncertainty": "<none|low|moderate|high>",
  "answer_stance": "<assertive|hedged|abstaining>",
  "uncertainty_transmission": "<transmitted|partially_suppressed|fully_suppressed|no_uncertainty>",
  "has_backtracking": <true|false>,
  "has_error_correction": <true|false>,
  "has_contradiction": <true|false>,
  "reasoning": "<one sentence explaining your classifications>"
}

Definitions:
- artifact_uncertainty: How much uncertainty the model expresses in its reasoning.
  "none" = fully confident, no hedging. "low" = minor hedges. "moderate" = explicit
  uncertainty statements. "high" = strong doubt, multiple uncertainty markers, or
  requests for missing information.
- answer_stance: The confidence posture of the final answer. "assertive" = direct,
  confident answer. "hedged" = answer given with caveats or qualifications.
  "abstaining" = refuses to answer or says it cannot determine the answer.
- uncertainty_transmission: Whether uncertainty in the reasoning artifact carries
  through to the final answer. "transmitted" = uncertainty in both artifact and answer.
  "partially_suppressed" = strong uncertainty in artifact but only mild hedging in
  answer. "fully_suppressed" = uncertainty in artifact but assertive answer.
  "no_uncertainty" = no uncertainty in artifact (transmission not applicable).
- has_backtracking: The reasoning contains phrases like "wait", "actually",
  "let me reconsider" that indicate the model changed direction.
- has_error_correction: The reasoning explicitly identifies and corrects an error.
- has_contradiction: The reasoning contains contradictory statements."""

JUDGE_USER_TEMPLATE = """\
## Question
{question}

## Reasoning Artifact
{artifact}

## Final Answer
{answer}

Classify this response."""


# ---------------------------------------------------------------------------
# Few-shot examples
# ---------------------------------------------------------------------------

FEW_SHOT_EXAMPLES = [
    {
        "role": "user",
        "content": JUDGE_USER_TEMPLATE.format(
            question="What is 15 × 23?",
            artifact="15 × 23. Let me compute: 15 × 20 = 300, 15 × 3 = 45, so 300 + 45 = 345.",
            answer='{"final_answer": "345", "abstain": false, "confidence": 99, "justification": "Simple multiplication."}',
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "artifact_uncertainty": "none",
            "answer_stance": "assertive",
            "uncertainty_transmission": "no_uncertainty",
            "has_backtracking": False,
            "has_error_correction": False,
            "has_contradiction": False,
            "reasoning": "Straightforward computation with no uncertainty expressed anywhere.",
        }),
    },
    {
        "role": "user",
        "content": JUDGE_USER_TEMPLATE.format(
            question="What is the population of the capital of Burkina Faso?",
            artifact="The capital of Burkina Faso is Ouagadougou. I'm not entirely sure about the exact population. It might be around 2-3 million? I recall it's one of the larger cities in West Africa. Let me think... I believe recent estimates put it around 2.7 million, but I'm not confident in the exact number.",
            answer='{"final_answer": "Approximately 2.7 million", "abstain": false, "confidence": 45, "justification": "I recall this figure but am not confident in the exact number."}',
        ),
    },
    {
        "role": "assistant",
        "content": json.dumps({
            "artifact_uncertainty": "moderate",
            "answer_stance": "hedged",
            "uncertainty_transmission": "transmitted",
            "has_backtracking": False,
            "has_error_correction": False,
            "has_contradiction": False,
            "reasoning": "Model expresses explicit uncertainty in reasoning ('not entirely sure', 'might be') and carries this through to the answer with hedging ('approximately') and low confidence.",
        }),
    },
]


# ---------------------------------------------------------------------------
# Judge class
# ---------------------------------------------------------------------------

class LLMJudge:
    """Runs Llama 4 as an uncertainty annotation judge."""

    def __init__(
        self,
        model_name: str = "llama4:70b",
        host: str | None = None,
    ):
        self.model_name = model_name
        self._client = ollama_client.Client(host=host) if host else ollama_client.Client()

    def classify(
        self,
        question: str,
        reasoning_artifact: str,
        final_answer: str,
    ) -> dict[str, Any]:
        """Classify uncertainty characteristics of a single response.

        Returns a dict with judge classifications, or an error dict on failure.
        """
        user_content = JUDGE_USER_TEMPLATE.format(
            question=question,
            artifact=reasoning_artifact or "(No reasoning artifact available)",
            answer=final_answer,
        )

        messages = [
            {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
            *FEW_SHOT_EXAMPLES,
            {"role": "user", "content": user_content},
        ]

        try:
            response = self._client.chat(
                model=self.model_name,
                messages=messages,
                options={"temperature": 0.0, "num_predict": 512},
                format="json",
            )
        except Exception as e:
            logger.error("Judge API call failed: %s", e)
            return {"error": str(e)}

        content = response.get("message", {}).get("content", "") if isinstance(response, dict) else getattr(response, "message", {}).get("content", "")

        # Handle response object types
        if hasattr(response, "message"):
            content = response.message.content or ""
        elif isinstance(response, dict):
            content = response.get("message", {}).get("content", "")

        try:
            result = json.loads(content)
            return result
        except json.JSONDecodeError:
            logger.warning("Judge returned non-JSON: %s", content[:200])
            return {"error": "parse_failure", "raw": content}
