"""Task suite curation script.

Downloads source datasets and creates the 500-prompt task suite by
random sampling from each bucket. Outputs JSONL files in data/tasks/.

Usage:
    python scripts/curate_tasks.py
    python scripts/curate_tasks.py --per-bucket 50   # smaller dev set
    python scripts/curate_tasks.py --seed 123         # different seed

Requirements:
    pip install datasets
"""

import argparse
import hashlib
import json
import logging
import random
import re
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.collect.base_collector import TaskBucket, TaskItem, QuestionMetadata

SEED = 42
PER_BUCKET = 100
TASKS_DIR = Path("data/tasks")


def make_prompt_id(bucket: str, idx: int, text: str) -> str:
    """Generate a deterministic prompt ID."""
    h = hashlib.md5(text.encode()).hexdigest()[:8]
    return f"{bucket}_{idx:04d}_{h}"


def write_tasks(tasks: list[TaskItem], bucket: TaskBucket, output_dir: Path = TASKS_DIR) -> None:
    """Write a list of TaskItems to the appropriate JSONL file."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{bucket.value}.jsonl"
    with open(path, "w") as f:
        for task in tasks:
            f.write(task.model_dump_json() + "\n")
    logger.info("Wrote %d tasks to %s", len(tasks), path)


# ---------------------------------------------------------------------------
# Bucket 1: Easy reasoning (GSM8K)
# ---------------------------------------------------------------------------

def curate_easy_reasoning(n: int, seed: int) -> list[TaskItem]:
    """Sample n easy reasoning tasks from GSM8K test set."""
    from datasets import load_dataset

    logger.info("Loading GSM8K...")
    ds = load_dataset("openai/gsm8k", "main", split="test")

    random.seed(seed)
    indices = random.sample(range(len(ds)), min(n, len(ds)))

    tasks = []
    for i, idx in enumerate(indices):
        row = ds[idx]
        question = row["question"]

        # Extract ground truth from answer (after #### separator)
        answer_text = row["answer"]
        gt = answer_text.split("####")[-1].strip() if "####" in answer_text else answer_text

        tasks.append(TaskItem(
            prompt_id=make_prompt_id("easy", i, question),
            question_text=question,
            task_bucket=TaskBucket.EASY_REASONING,
            metadata=QuestionMetadata(
                source_dataset="gsm8k",
                ground_truth=gt,
                is_answerable=True,
                difficulty="easy",
            ),
        ))

    return tasks


# ---------------------------------------------------------------------------
# Bucket 2: Hard reasoning (MATH Level 4-5)
# ---------------------------------------------------------------------------

def curate_hard_reasoning(n: int, seed: int) -> list[TaskItem]:
    """Sample n hard reasoning tasks from MATH dataset (Level 4-5)."""
    from datasets import load_dataset

    logger.info("Loading MATH dataset...")

    # Try multiple HF sources for the MATH dataset
    # EleutherAI version requires loading each subject config separately
    MATH_CONFIGS = [
        "algebra", "counting_and_probability", "geometry",
        "intermediate_algebra", "number_theory", "prealgebra", "precalculus",
    ]

    ds = None

    # Try 1: hendrycks original (often gated)
    try:
        ds = load_dataset("hendrycks/competition_math", split="test")
        logger.info("Loaded MATH from hendrycks/competition_math")
    except Exception as e:
        logger.warning("Could not load hendrycks/competition_math: %s", e)

    # Try 2: EleutherAI version with subject configs
    if ds is None:
        try:
            all_items = []
            for config in MATH_CONFIGS:
                subset = load_dataset("EleutherAI/hendrycks_math", config, split="test")
                all_items.extend(list(subset))
            # Convert to a simple list-based dataset
            from datasets import Dataset
            ds = Dataset.from_list(all_items)
            logger.info("Loaded MATH from EleutherAI/hendrycks_math (%d items across %d configs)", len(ds), len(MATH_CONFIGS))
        except Exception as e:
            logger.warning("Could not load EleutherAI/hendrycks_math: %s", e)

    if ds is None:
        logger.error("Could not load MATH from any source. Using GSM8K hard subset as fallback.")
        return _curate_hard_gsm8k_fallback(n, seed)

    # Detect level column name
    level_col = None
    for col in ["level", "Level", "difficulty"]:
        if col in ds.column_names:
            level_col = col
            break

    # Detect problem column name
    problem_col = None
    for col in ["problem", "question", "input"]:
        if col in ds.column_names:
            problem_col = col
            break

    # Detect solution column name
    solution_col = None
    for col in ["solution", "answer", "output"]:
        if col in ds.column_names:
            solution_col = col
            break

    if problem_col is None:
        logger.error("Could not find problem column in MATH dataset. Columns: %s", ds.column_names)
        return _curate_hard_gsm8k_fallback(n, seed)

    # Filter to Level 4 and Level 5
    if level_col:
        hard = [row for row in ds if row[level_col] in ("Level 4", "Level 5", 4, 5)]
    else:
        # No level info — take all problems
        hard = list(ds)
        logger.warning("No level column found; using all %d problems", len(hard))

    logger.info("Found %d Level 4-5 problems", len(hard))

    random.seed(seed)
    selected = random.sample(hard, min(n, len(hard)))

    tasks = []
    for i, row in enumerate(selected):
        question = row[problem_col]

        # Extract ground truth from \boxed{...}
        solution_text = row.get(solution_col, "") if solution_col else ""
        gt_match = re.search(r"\\boxed\{(.+?)\}", solution_text)
        gt = gt_match.group(1) if gt_match else solution_text[-50:] if solution_text else ""

        difficulty = row.get(level_col, "") if level_col else ""

        tasks.append(TaskItem(
            prompt_id=make_prompt_id("hard", i, question),
            question_text=question,
            task_bucket=TaskBucket.HARD_REASONING,
            metadata=QuestionMetadata(
                source_dataset="math",
                ground_truth=gt,
                is_answerable=True,
                difficulty=str(difficulty),
            ),
        ))

    return tasks


def _curate_hard_gsm8k_fallback(n: int, seed: int) -> list[TaskItem]:
    """Fallback: use GSM8K train set (harder problems by answer magnitude)."""
    from datasets import load_dataset

    logger.info("Fallback: loading GSM8K train for hard reasoning...")
    ds = load_dataset("openai/gsm8k", "main", split="train")

    # Use problems with longer solutions as a proxy for difficulty
    items = list(ds)
    items.sort(key=lambda x: len(x["answer"]), reverse=True)
    hard_subset = items[:n * 3]  # top third by solution length

    random.seed(seed)
    selected = random.sample(hard_subset, min(n, len(hard_subset)))

    tasks = []
    for i, row in enumerate(selected):
        question = row["question"]
        gt = row["answer"].split("####")[-1].strip() if "####" in row["answer"] else row["answer"]

        tasks.append(TaskItem(
            prompt_id=make_prompt_id("hard", i, question),
            question_text=question,
            task_bucket=TaskBucket.HARD_REASONING,
            metadata=QuestionMetadata(
                source_dataset="gsm8k_hard",
                ground_truth=gt,
                is_answerable=True,
                difficulty="hard",
            ),
        ))

    return tasks


# ---------------------------------------------------------------------------
# Bucket 3: Unanswerable (SQuAD v2 + BigBench Known Unknowns)
# ---------------------------------------------------------------------------

def curate_unanswerable(n: int, seed: int) -> list[TaskItem]:
    """Sample n unanswerable questions from SQuAD v2 and BigBench Known Unknowns.

    Sources:
    - SQuAD v2 (rajpurkar/squad_v2): reading comprehension with unanswerable
      questions. We present the question WITHOUT context so the model must
      recognize it lacks the information to answer.
    - BigBench Known Unknowns (tasksource/bigbench, known_unknowns): questions
      designed to be unknowable. Small (30 items) but high quality.
    """
    from datasets import load_dataset

    all_items: list[dict] = []

    # Source 1: SQuAD v2 unanswerable questions
    logger.info("Loading SQuAD v2 unanswerable questions...")
    try:
        squad = load_dataset("rajpurkar/squad_v2", split="validation")
        unanswerable = [
            {"question": row["question"], "source": "squad_v2"}
            for row in squad
            if len(row["answers"]["text"]) == 0
        ]
        logger.info("Found %d unanswerable questions in SQuAD v2", len(unanswerable))
        all_items.extend(unanswerable)
    except Exception as e:
        logger.error("Could not load SQuAD v2: %s", e)

    # Source 2: BigBench Known Unknowns
    logger.info("Loading BigBench Known Unknowns...")
    try:
        bb = load_dataset("tasksource/bigbench", "known_unknowns", split="train")
        for row in bb:
            # Extract question from the 'inputs' field (format: "Q: ...\n  choice: ...\nA:")
            text = row["inputs"]
            q_match = re.match(r"Q:\s*(.+?)(?:\n|$)", text)
            question = q_match.group(1).strip() if q_match else text
            all_items.append({"question": question, "source": "bigbench_known_unknowns"})
        logger.info("Added %d items from BigBench Known Unknowns", len(bb))
    except Exception as e:
        logger.warning("Could not load BigBench Known Unknowns: %s", e)

    if not all_items:
        logger.error("No unanswerable items loaded from any source")
        return []

    logger.info("Total unanswerable pool: %d items", len(all_items))

    random.seed(seed)
    selected = random.sample(all_items, min(n, len(all_items)))

    tasks = []
    for i, item in enumerate(selected):
        question = item["question"]
        tasks.append(TaskItem(
            prompt_id=make_prompt_id("unans", i, question),
            question_text=question,
            task_bucket=TaskBucket.UNANSWERABLE,
            metadata=QuestionMetadata(
                source_dataset=item["source"],
                ground_truth=None,
                is_answerable=False,
                difficulty=None,
            ),
        ))

    return tasks


# ---------------------------------------------------------------------------
# Bucket 4: Underspecified (CoCoNot + BBQ ambiguous)
# ---------------------------------------------------------------------------

def curate_underspecified(n: int, seed: int) -> list[TaskItem]:
    """Sample n underspecified/ambiguous questions from CoCoNot and BBQ.

    Sources:
    - CoCoNot (allenai/coconot, 'original'): requests that are incomplete,
      indeterminate, or unsupported. We use the 'Incomplete requests' and
      'Indeterminate requests' categories.
    - BBQ (lighteval/bbq_helm): bias benchmark with ambiguous contexts where
      the correct answer is "Cannot be determined". We sample across multiple
      social categories.
    """
    from datasets import load_dataset

    all_items: list[dict] = []

    # Source 1: CoCoNot incomplete + indeterminate requests
    logger.info("Loading CoCoNot for underspecified questions...")
    try:
        coconot = load_dataset("allenai/coconot", "original", split="test")
        target_categories = {"Incomplete requests", "Indeterminate requests", "Unsupported requests"}
        for row in coconot:
            if row["category"] in target_categories:
                all_items.append({
                    "question": row["prompt"],
                    "source": "coconot",
                    "subcategory": row.get("subcategory", row["category"]),
                })
        logger.info("Found %d CoCoNot items (incomplete/indeterminate/unsupported)", len(all_items))
    except Exception as e:
        logger.error("Could not load CoCoNot: %s", e)

    # Source 2: BBQ ambiguous contexts (correct answer = "Cannot be determined")
    logger.info("Loading BBQ for ambiguous questions...")
    bbq_categories = ["Age", "Gender_identity", "Nationality", "Race_ethnicity", "Religion"]
    bbq_count = 0
    for category in bbq_categories:
        try:
            bbq = load_dataset("lighteval/bbq_helm", category, split="test")
            # Select items where gold answer is "Cannot be determined" (index 1 typically)
            for row in bbq:
                if row["gold_index"] < len(row["choices"]) and "cannot be determined" in row["choices"][row["gold_index"]].lower():
                    question = f"{row['context']} {row['question']}"
                    all_items.append({
                        "question": question,
                        "source": "bbq",
                        "subcategory": category,
                    })
                    bbq_count += 1
        except Exception as e:
            logger.warning("Could not load BBQ category %s: %s", category, e)

    logger.info("Found %d BBQ ambiguous items", bbq_count)

    if not all_items:
        logger.error("No underspecified items loaded from any source")
        return []

    logger.info("Total underspecified pool: %d items", len(all_items))

    random.seed(seed)
    selected = random.sample(all_items, min(n, len(all_items)))

    tasks = []
    for i, item in enumerate(selected):
        question = item["question"]
        tasks.append(TaskItem(
            prompt_id=make_prompt_id("under", i, question),
            question_text=question,
            task_bucket=TaskBucket.UNDERSPECIFIED,
            metadata=QuestionMetadata(
                source_dataset=item["source"],
                ground_truth=None,
                is_answerable=False,
                difficulty=item.get("subcategory"),
            ),
        ))

    return tasks


# ---------------------------------------------------------------------------
# Bucket 5: Factual QA (TriviaQA)
# ---------------------------------------------------------------------------

def curate_factual_qa(n: int, seed: int) -> list[TaskItem]:
    """Sample n factual QA tasks from TriviaQA validation set."""
    from datasets import load_dataset

    logger.info("Loading TriviaQA...")
    ds = load_dataset("trivia_qa", "rc.nocontext", split="validation")

    random.seed(seed)
    indices = random.sample(range(len(ds)), min(n, len(ds)))

    tasks = []
    for i, idx in enumerate(indices):
        row = ds[idx]
        question = row["question"]

        # Ground truth: primary answer value
        answer = row.get("answer", {})
        gt = answer.get("value", "") if isinstance(answer, dict) else str(answer)

        tasks.append(TaskItem(
            prompt_id=make_prompt_id("fact", i, question),
            question_text=question,
            task_bucket=TaskBucket.FACTUAL_QA,
            metadata=QuestionMetadata(
                source_dataset="triviaqa",
                ground_truth=gt,
                is_answerable=True,
                difficulty=None,
            ),
        ))

    return tasks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CURATORS = {
    TaskBucket.EASY_REASONING: curate_easy_reasoning,
    TaskBucket.HARD_REASONING: curate_hard_reasoning,
    TaskBucket.UNANSWERABLE: curate_unanswerable,
    TaskBucket.UNDERSPECIFIED: curate_underspecified,
    TaskBucket.FACTUAL_QA: curate_factual_qa,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Curate the 500-prompt task suite")
    parser.add_argument("--per-bucket", type=int, default=PER_BUCKET,
                        help="Number of prompts per bucket (default: 100)")
    parser.add_argument("--seed", type=int, default=SEED,
                        help="Random seed for reproducibility")
    parser.add_argument("--output-dir", type=Path, default=TASKS_DIR)
    parser.add_argument("--buckets", nargs="+", default=None,
                        help="Only curate specific buckets (e.g. easy_reasoning factual_qa)")
    parser.add_argument("--log-level", type=str, default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_dir = args.output_dir

    # Determine which buckets to curate
    if args.buckets:
        buckets = [TaskBucket(b) for b in args.buckets]
    else:
        buckets = list(TaskBucket)

    total = 0
    for bucket in buckets:
        curator = CURATORS.get(bucket)
        if curator is None:
            logger.warning("No curator for bucket '%s'", bucket.value)
            continue

        print(f"\n--- Curating: {bucket.value} ({args.per_bucket} items) ---")
        tasks = curator(args.per_bucket, args.seed)
        write_tasks(tasks, bucket, output_dir=output_dir)
        total += len(tasks)

    print(f"\n=== Curation Complete: {total} tasks across {len(buckets)} buckets ===")
    print(f"Output: {output_dir}/")


if __name__ == "__main__":
    main()
