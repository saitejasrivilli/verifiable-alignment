"""
training/constitutional_ai.py

Constitutional AI: model critiques its own output against a list of principles,
then revises. Generates (original, revised) pairs that can feed DPO training.

Pipeline:
  1. Sample model response to a prompt
  2. For each principle in the constitution, ask the judge model:
     "Does this response violate principle X? If so, rewrite it."
  3. Accept revision if the critique identifies a violation
  4. Build (original, revised) preference pair: revised=chosen, original=rejected
  5. Save as preference_pairs.jsonl for DPO training

Constitution (12 principles focused on math/reasoning quality):
  - Correctness:       solution is mathematically correct
  - Step clarity:      each reasoning step is explicit and follows from the previous
  - No hallucination: no invented facts or unjustified leaps
  - Format:            answer is boxed with \\boxed{}
  - Conciseness:       no unnecessary repetition
  - No circular:       conclusion is not used as a premise
  - Units:             physical quantities include correct units
  - Edge cases:        solution handles boundary conditions
  - Verification:      solution includes a sanity check
  - Notation:          mathematical notation is standard and unambiguous
  - No hedging:        claims are stated with appropriate confidence
  - Completeness:      all required steps shown, nothing skipped

Usage:
    # Use Groq (fast, no local GPU)
    python constitutional_ai.py \\
        --judge groq --groq_model llama-3.1-8b-instant \\
        --n 200 --output data/constitutional_pairs.jsonl

    # Use local model as judge
    python constitutional_ai.py \\
        --judge local --judge_model Qwen/Qwen2.5-7B-Instruct \\
        --n 200 --output data/constitutional_pairs.jsonl

    # Load pre-generated responses and skip generation
    python constitutional_ai.py \\
        --judge groq --input_responses data/cached_responses.jsonl \\
        --n 200 --output data/constitutional_pairs.jsonl
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Constitution ──────────────────────────────────────────────────────────────
@dataclass
class Principle:
    name: str
    critique_prompt: str   # "Does this response [violate X]? Answer YES or NO, then explain."
    revision_prompt: str   # "Rewrite the response to [fix X]."


CONSTITUTION: list[Principle] = [
    Principle(
        name="correctness",
        critique_prompt=(
            "Does this math response contain a mathematical error — wrong calculation, "
            "incorrect formula, or wrong final answer? Answer YES or NO, then explain briefly."
        ),
        revision_prompt=(
            "Rewrite the response to fix all mathematical errors. "
            "Show correct calculations and arrive at the right answer."
        ),
    ),
    Principle(
        name="step_clarity",
        critique_prompt=(
            "Does this response skip reasoning steps or make jumps that are not justified "
            "by previous steps? Answer YES or NO, then explain briefly."
        ),
        revision_prompt=(
            "Rewrite the response so every reasoning step is explicit, "
            "clearly labeled, and follows logically from the previous step."
        ),
    ),
    Principle(
        name="no_hallucination",
        critique_prompt=(
            "Does this response introduce facts, formulas, or values that are not "
            "given in the problem and not standard mathematical knowledge? "
            "Answer YES or NO, then explain briefly."
        ),
        revision_prompt=(
            "Rewrite the response using only the information given in the problem "
            "and established mathematical facts. Remove all unjustified leaps."
        ),
    ),
    Principle(
        name="format",
        critique_prompt=(
            "Does this response fail to box the final answer with \\boxed{}? "
            "Answer YES or NO, then explain briefly."
        ),
        revision_prompt=(
            "Rewrite the response and box the final answer using \\boxed{} notation."
        ),
    ),
    Principle(
        name="conciseness",
        critique_prompt=(
            "Does this response repeat the same information, restate steps already shown, "
            "or include unnecessary filler text? Answer YES or NO, then explain briefly."
        ),
        revision_prompt=(
            "Rewrite the response removing all repetition and unnecessary filler "
            "while keeping every logically necessary step."
        ),
    ),
    Principle(
        name="no_circular_reasoning",
        critique_prompt=(
            "Does this response use the conclusion as a premise, or assume what it "
            "is trying to prove? Answer YES or NO, then explain briefly."
        ),
        revision_prompt=(
            "Rewrite the response to eliminate circular reasoning. "
            "Derive the conclusion only from the given premises and established facts."
        ),
    ),
    Principle(
        name="units",
        critique_prompt=(
            "Does this response involve physical quantities but omit or use incorrect units? "
            "Answer YES or NO, then explain briefly."
        ),
        revision_prompt=(
            "Rewrite the response ensuring every physical quantity carries the correct unit "
            "and unit conversions are performed explicitly where needed."
        ),
    ),
    Principle(
        name="edge_cases",
        critique_prompt=(
            "Does this response fail to handle boundary conditions — "
            "such as zero, negative values, empty sets, or division by zero? "
            "Answer YES or NO, then explain briefly."
        ),
        revision_prompt=(
            "Rewrite the response to explicitly address all relevant boundary conditions "
            "and explain how the solution handles them."
        ),
    ),
    Principle(
        name="verification",
        critique_prompt=(
            "Does this response lack a sanity check — e.g., substituting the answer "
            "back into the original equation, checking units, or estimating the order of magnitude? "
            "Answer YES or NO, then explain briefly."
        ),
        revision_prompt=(
            "Rewrite the response and add a verification step that confirms the answer "
            "is correct, such as back-substitution or a numerical check."
        ),
    ),
    Principle(
        name="notation",
        critique_prompt=(
            "Does this response use non-standard, ambiguous, or inconsistent "
            "mathematical notation? Answer YES or NO, then explain briefly."
        ),
        revision_prompt=(
            "Rewrite the response using standard, unambiguous mathematical notation "
            "consistently throughout."
        ),
    ),
    Principle(
        name="no_hedging",
        critique_prompt=(
            "Does this response hedge unnecessarily — saying 'I think', 'maybe', "
            "'it could be' — when the mathematics leads to a definite conclusion? "
            "Answer YES or NO, then explain briefly."
        ),
        revision_prompt=(
            "Rewrite the response stating all conclusions with appropriate mathematical "
            "confidence. Remove hedging language where the answer is provably correct."
        ),
    ),
    Principle(
        name="completeness",
        critique_prompt=(
            "Does this response omit required steps — glossing over a key algebraic "
            "manipulation, skipping a case, or jumping to the answer without proof? "
            "Answer YES or NO, then explain briefly."
        ),
        revision_prompt=(
            "Rewrite the response showing every required step. Nothing should be "
            "left as an exercise or implied without justification."
        ),
    ),
]


# ── Judge backends ────────────────────────────────────────────────────────────
class GroqJudge:
    """AI judge backed by the Groq API. Requires GROQ_API_KEY env var."""

    def __init__(self, model: str = "llama-3.1-8b-instant"):
        try:
            from groq import Groq
        except ImportError as e:
            raise ImportError("pip install groq") from e

        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError("GROQ_API_KEY env var not set")

        self.client = Groq(api_key=api_key)
        self.model = model

    def judge(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=512,
            temperature=0.0,
        )
        return response.choices[0].message.content.strip()


class LocalJudge:
    """AI judge backed by a locally loaded HuggingFace model."""

    def __init__(self, model_name: str):
        logger.info(f"Loading local judge model: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        self.model.eval()

    @torch.no_grad()
    def judge(self, system: str, user: str) -> str:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = self.tokenizer(text, return_tensors="pt").to(
            next(self.model.parameters()).device
        )
        outputs = self.model.generate(
            **enc,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_tokens = outputs[0][enc["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# ── Critique loop ─────────────────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "You are a rigorous mathematical reasoning evaluator. "
    "Be precise, objective, and concise in your assessments."
)


def run_critique_loop(
    judge,
    prompt: str,
    response: str,
    principles: list[Principle],
    max_principles: int = 3,
) -> tuple[Optional[str], Optional[str]]:
    """
    Iterate over up to max_principles. Stop at first violation found.

    Returns:
        (revised_response, violated_principle_name) if a violation was found.
        (None, None) if no violation detected.
    """
    for principle in principles[:max_principles]:
        user_msg = (
            f"ORIGINAL PROBLEM:\n{prompt}\n\n"
            f"MODEL RESPONSE:\n{response}\n\n"
            f"CRITIQUE QUESTION:\n{principle.critique_prompt}"
        )
        critique = judge.judge(system=SYSTEM_PROMPT, user=user_msg)

        # Detect YES — case-insensitive, first word check
        first_word = critique.strip().split()[0].upper().strip(".,:")
        if first_word == "YES":
            revision_msg = (
                f"ORIGINAL PROBLEM:\n{prompt}\n\n"
                f"CURRENT RESPONSE:\n{response}\n\n"
                f"REVISION INSTRUCTION:\n{principle.revision_prompt}"
            )
            revised = judge.judge(system=SYSTEM_PROMPT, user=revision_msg)
            return revised, principle.name

    return None, None


# ── Response generation ───────────────────────────────────────────────────────
def generate_responses(
    model_name: str,
    prompts: list[str],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
) -> list[str]:
    """Generate one response per prompt using a local policy model."""
    logger.info(f"Loading policy model for generation: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    responses = []
    device = next(model.parameters()).device

    for i, prompt in enumerate(prompts):
        if i % 50 == 0:
            logger.info(f"Generating response {i}/{len(prompts)}")
        enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = out[0][enc["input_ids"].shape[1]:]
        responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())

    return responses


# ── Dataset loading ───────────────────────────────────────────────────────────
def load_prompts(n: int, dataset_id: str = "lighteval/MATH", split: str = "train") -> list[str]:
    ds = load_dataset(dataset_id, split=split)
    ds = ds.select(range(min(n, len(ds))))

    def format_prompt(ex):
        return (
            "Solve the following math problem step by step. "
            "Box your final answer with \\boxed{}.\n\n"
            f"Problem: {ex['problem']}\n\nSolution:"
        )

    return [format_prompt(ex) for ex in ds]


# ── Main pipeline ─────────────────────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:
    wandb.init(
        project="verifiable-alignment",
        name=f"constitutional-ai-{args.judge}",
        config=vars(args),
    )

    # ── Build judge ───────────────────────────────────────────────────────────
    if args.judge == "groq":
        judge = GroqJudge(model=args.groq_model)
    else:
        judge = LocalJudge(model_name=args.judge_model)

    # ── Load prompts ──────────────────────────────────────────────────────────
    prompts = load_prompts(args.n)
    logger.info(f"Loaded {len(prompts)} prompts")

    # ── Load or generate responses ────────────────────────────────────────────
    if args.input_responses:
        logger.info(f"Loading cached responses from {args.input_responses}")
        responses = []
        with open(args.input_responses) as f:
            for line in f:
                item = json.loads(line)
                responses.append(item["response"])
        responses = responses[: len(prompts)]
    else:
        responses = generate_responses(
            model_name=args.policy_model,
            prompts=prompts,
        )

    # ── Critique + revision loop ──────────────────────────────────────────────
    preference_pairs: list[dict] = []
    violation_counts: dict[str, int] = defaultdict(int)
    lengths_before: list[int] = []
    lengths_after: list[int] = []
    n_revised = 0

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as out_f:
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            if i % 20 == 0:
                logger.info(f"Processing example {i}/{len(prompts)}")

            revised, violated_principle = run_critique_loop(
                judge=judge,
                prompt=prompt,
                response=response,
                principles=CONSTITUTION,
                max_principles=args.max_principles,
            )

            lengths_before.append(len(response.split()))

            if revised is not None:
                n_revised += 1
                violation_counts[violated_principle] += 1
                lengths_after.append(len(revised.split()))
                pair = {
                    "prompt": prompt,
                    "chosen": revised,
                    "rejected": response,
                }
                preference_pairs.append(pair)
                out_f.write(json.dumps(pair) + "\n")
            else:
                lengths_after.append(len(response.split()))

    # ── Metrics ───────────────────────────────────────────────────────────────
    revision_rate = n_revised / max(len(prompts), 1)
    avg_len_before = sum(lengths_before) / max(len(lengths_before), 1)
    avg_len_after = sum(lengths_after) / max(len(lengths_after), 1)

    logger.info(f"Revision rate:           {revision_rate:.2%} ({n_revised}/{len(prompts)})")
    logger.info(f"Avg response length:     {avg_len_before:.1f} tokens → {avg_len_after:.1f} tokens")
    logger.info("Per-principle violation rates:")
    for name, count in sorted(violation_counts.items(), key=lambda x: -x[1]):
        logger.info(f"  {name:<25} {count / max(len(prompts), 1):.2%}")

    wandb.log(
        {
            "constitutional/revision_rate": revision_rate,
            "constitutional/pairs_generated": len(preference_pairs),
            "constitutional/avg_len_before": avg_len_before,
            "constitutional/avg_len_after": avg_len_after,
            **{
                f"constitutional/violation_rate/{name}": count / max(len(prompts), 1)
                for name, count in violation_counts.items()
            },
        }
    )

    logger.info(f"Preference pairs saved to {output_path} ({len(preference_pairs)} pairs)")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--judge",
        choices=["groq", "local"],
        default="groq",
        help="Judge backend to use for critiques and revisions.",
    )
    parser.add_argument(
        "--groq_model",
        default="llama-3.1-8b-instant",
        help="Groq model id (used when --judge groq).",
    )
    parser.add_argument(
        "--judge_model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="HuggingFace model name for local judge.",
    )
    parser.add_argument(
        "--policy_model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model used to generate initial responses (ignored if --input_responses set).",
    )
    parser.add_argument(
        "--input_responses",
        default=None,
        help="Path to a JSONL file with pre-generated responses {prompt, response}. "
             "Skips the generation step.",
    )
    parser.add_argument("--n", type=int, default=200, help="Number of prompts to process.")
    parser.add_argument(
        "--max_principles",
        type=int,
        default=3,
        help="Max principles to check per response (stops at first violation).",
    )
    parser.add_argument(
        "--output",
        default="data/constitutional_pairs.jsonl",
        help="Output path for preference pairs (DPO-compatible JSONL).",
    )
    main(parser.parse_args())
