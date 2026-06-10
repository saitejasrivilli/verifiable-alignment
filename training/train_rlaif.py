"""
training/train_rlaif.py

RLAIF: RL from AI Feedback — generates preference labels using an AI judge
at scale, then trains with DPO. No human labelers required.

Pipeline:
  1. Sample K completions per prompt from the policy model (at temperature 0.7)
  2. Score each completion with an AI judge (Groq or local model)
  3. Form (best, worst) preference pairs from the ranked completions
  4. Train DPO on the AI-generated preference pairs
  5. Evaluate: MATH accuracy before vs. after

Judge prompt for math:
  "Score this math solution from 1-10 on correctness, clarity, and format.
   Return ONLY a JSON: {score: N, correct: true/false, reasoning: '...'}"

This covers the RLAIF case where verifiable rewards are not available —
the AI judge provides the preference signal instead of a symbolic verifier.
Contrast with RLVR (train_grpo.py) where sympy verifies correctness exactly.

Comparison of alignment approaches:

  Method             Label source    Scale           Cost
  -----------------  --------------  --------------  ---------------
  RLHF               Humans          Limited (~1K)   High
  RLAIF              AI judge        Unlimited       Low (API)
  RLVR               Symbolic        Unlimited       Zero (local)
  Constitutional AI  Self            Unlimited       Zero (local)

Usage:
    # Generate AI-labeled preference pairs
    python train_rlaif.py --stage label \\
        --model Qwen/Qwen2.5-7B-Instruct \\
        --judge groq --n_prompts 500 --k_completions 4 \\
        --output data/rlaif_pairs.jsonl

    # Train DPO on AI-labeled pairs
    python train_rlaif.py --stage train \\
        --pairs data/rlaif_pairs.jsonl \\
        --output_dir checkpoints/rlaif_dpo

    # Full pipeline: label + train
    python train_rlaif.py --stage both \\
        --model Qwen/Qwen2.5-7B-Instruct --judge groq
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
import wandb
from datasets import Dataset, load_dataset
from omegaconf import OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from trl import DPOConfig, DPOTrainer

from data.verifier import MathVerifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

VERIFIER = MathVerifier()

# Minimum score gap between chosen and rejected to keep a pair
MIN_SCORE_GAP = 2.0


# ── Judge score container ─────────────────────────────────────────────────────
@dataclass
class JudgeScore:
    score: float          # 1–10 composite score
    correct: bool         # judge believes the answer is correct
    reasoning: str        # brief explanation


# ── Judge backends ────────────────────────────────────────────────────────────
JUDGE_SYSTEM = (
    "You are a rigorous math evaluator. "
    "Score the solution and return ONLY valid JSON — no extra text."
)

JUDGE_USER_TMPL = (
    "PROBLEM:\n{problem}\n\n"
    "SOLUTION:\n{solution}\n\n"
    "Score this math solution from 1-10 on correctness, clarity, and format. "
    'Return ONLY a JSON object: {{"score": N, "correct": true/false, "reasoning": "..."}}'
)


def _parse_judge_json(raw: str) -> Optional[JudgeScore]:
    """Extract JSON from judge output, tolerating markdown fences."""
    import re

    raw = raw.strip()
    # Strip markdown code fences if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        obj = json.loads(raw)
        return JudgeScore(
            score=float(obj.get("score", 5)),
            correct=bool(obj.get("correct", False)),
            reasoning=str(obj.get("reasoning", "")),
        )
    except (json.JSONDecodeError, KeyError, TypeError):
        logger.debug(f"Failed to parse judge JSON: {raw[:200]}")
        return None


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

    def score(self, problem: str, solution: str) -> Optional[JudgeScore]:
        user_msg = JUDGE_USER_TMPL.format(problem=problem, solution=solution)
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": user_msg},
                ],
                max_tokens=256,
                temperature=0.0,
            )
            return _parse_judge_json(resp.choices[0].message.content)
        except Exception as e:
            logger.warning(f"Groq judge error: {e}")
            return None


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
    def score(self, problem: str, solution: str) -> Optional[JudgeScore]:
        user_msg = JUDGE_USER_TMPL.format(problem=problem, solution=solution)
        messages = [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_msg},
        ]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = self.tokenizer(text, return_tensors="pt").to(
            next(self.model.parameters()).device
        )
        out = self.model.generate(
            **enc,
            max_new_tokens=256,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        new_tokens = out[0][enc["input_ids"].shape[1]:]
        raw = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return _parse_judge_json(raw)


# ── Dataset helpers ───────────────────────────────────────────────────────────
def load_math_prompts(
    n: int,
    dataset_id: str = "lighteval/MATH",
    split: str = "train",
) -> list[dict]:
    """Returns list of {prompt, ground_truth} dicts."""
    ds = load_dataset(dataset_id, split=split)
    ds = ds.select(range(min(n, len(ds))))

    records = []
    for ex in ds:
        records.append(
            {
                "prompt": (
                    "Solve the following math problem step by step. "
                    "Box your final answer with \\boxed{}.\n\n"
                    f"Problem: {ex['problem']}\n\nSolution:"
                ),
                "ground_truth": ex.get("solution", ""),
            }
        )
    return records


# ── Rollout generation ────────────────────────────────────────────────────────
def generate_k_completions(
    model_name: str,
    prompts: list[str],
    k: int = 4,
    temperature: float = 0.7,
    max_new_tokens: int = 512,
) -> list[list[str]]:
    """
    For each prompt, generate K completions at the given temperature.
    Returns a list of lists: completions[i] = k completions for prompts[i].
    """
    logger.info(f"Loading policy model for rollouts: {model_name}")
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
    device = next(model.parameters()).device

    all_completions: list[list[str]] = []

    for i, prompt in enumerate(prompts):
        if i % 50 == 0:
            logger.info(f"Rolling out prompt {i}/{len(prompts)}")

        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(device)

        completions = []
        for _ in range(k):
            with torch.no_grad():
                out = model.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=temperature,
                    pad_token_id=tokenizer.pad_token_id,
                )
            new_tokens = out[0][enc["input_ids"].shape[1]:]
            completions.append(tokenizer.decode(new_tokens, skip_special_tokens=True).strip())

        all_completions.append(completions)

    return all_completions


# ── Labeling stage ────────────────────────────────────────────────────────────
def label_stage(args: argparse.Namespace) -> Path:
    """
    Score K completions per prompt with the AI judge.
    Form preference pairs: highest-scored = chosen, lowest-scored = rejected.
    Filter pairs where score gap < MIN_SCORE_GAP.
    """
    records = load_math_prompts(args.n_prompts)
    prompts = [r["prompt"] for r in records]
    ground_truths = [r["ground_truth"] for r in records]

    # Generate K completions per prompt
    all_completions = generate_k_completions(
        model_name=args.model,
        prompts=prompts,
        k=args.k_completions,
        temperature=0.7,
    )

    # Build judge
    if args.judge == "groq":
        judge = GroqJudge(model=args.groq_model)
    else:
        judge = LocalJudge(model_name=args.judge_model)

    output_path = Path(args.pairs if args.pairs else args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    pairs_generated = 0
    pairs_filtered = 0
    score_gaps: list[float] = []
    judge_correct_with_verifier = 0
    verifier_checks = 0

    with open(output_path, "w") as out_f:
        for i, (prompt, completions, gt) in enumerate(
            zip(prompts, all_completions, ground_truths)
        ):
            if i % 20 == 0:
                logger.info(f"Scoring prompt {i}/{len(prompts)}")

            scores: list[tuple[float, str, Optional[JudgeScore]]] = []
            for completion in completions:
                judge_result = judge.score(problem=prompt, solution=completion)
                sc = judge_result.score if judge_result is not None else 5.0
                scores.append((sc, completion, judge_result))

                # Check judge agreement with symbolic verifier
                if gt and judge_result is not None:
                    verifier_says_correct = VERIFIER.verify(completion, gt)
                    if judge_result.correct == verifier_says_correct:
                        judge_correct_with_verifier += 1
                    verifier_checks += 1

            scores.sort(key=lambda x: x[0], reverse=True)
            best_score, chosen, _ = scores[0]
            worst_score, rejected, _ = scores[-1]
            gap = best_score - worst_score

            pairs_generated += 1
            if gap < MIN_SCORE_GAP:
                pairs_filtered += 1
                continue

            score_gaps.append(gap)
            pair = {"prompt": prompt, "chosen": chosen, "rejected": rejected}
            out_f.write(json.dumps(pair) + "\n")

    kept = pairs_generated - pairs_filtered
    avg_gap = sum(score_gaps) / max(len(score_gaps), 1)
    judge_agree = judge_correct_with_verifier / max(verifier_checks, 1)

    logger.info(f"Pairs generated:          {pairs_generated}")
    logger.info(f"Pairs filtered (gap<{MIN_SCORE_GAP}): {pairs_filtered}")
    logger.info(f"Pairs kept:               {kept}")
    logger.info(f"Avg score gap:            {avg_gap:.2f}")
    logger.info(f"Judge/verifier agreement: {judge_agree:.2%}")

    wandb.log(
        {
            "rlaif/pairs_generated": pairs_generated,
            "rlaif/pairs_filtered": pairs_filtered,
            "rlaif/pairs_kept": kept,
            "rlaif/avg_score_gap": avg_gap,
            "rlaif/judge_agree_with_verifier": judge_agree,
        }
    )

    logger.info(f"RLAIF pairs saved to {output_path}")
    return output_path


# ── DPO training (mirrors train_dpo.py setup) ─────────────────────────────────
class LengthMonitorCallback(TrainerCallback):
    """Alert if mean response length drops >20% — early reward hacking signal."""

    def __init__(self):
        self.baseline_length: Optional[float] = None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "train/chosen_response_length" in logs:
            length = logs["train/chosen_response_length"]
            if self.baseline_length is None:
                self.baseline_length = length
            elif length < self.baseline_length * 0.8:
                logger.warning(
                    f"Length collapse detected at step {state.global_step}: "
                    f"{length:.0f} vs baseline {self.baseline_length:.0f}"
                )
                wandb.alert(
                    title="Length Collapse",
                    text=f"Mean chosen length dropped to {length:.0f} "
                         f"({100 * length / self.baseline_length:.0f}% of baseline)",
                    level=wandb.AlertLevel.WARN,
                )


def load_model_and_tokenizer(model_name: str, lora_r: int = 16, lora_alpha: int = 32):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # DPO requires left-padding

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )

    # Reference model: separate copy, frozen
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
        device_map="auto",
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # Apply LoRA — same config as train_dpo.py (β=0.1)
    lora_cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    return model, ref_model, tokenizer


def train_stage(args: argparse.Namespace) -> None:
    """Run DPO on AI-labeled preference pairs, reusing train_dpo.py setup."""
    pairs_path = Path(args.pairs)
    if not pairs_path.exists():
        raise FileNotFoundError(f"Pairs file not found: {pairs_path}")

    records = []
    with open(pairs_path) as f:
        for line in f:
            records.append(json.loads(line))

    logger.info(f"Loaded {len(records)} RLAIF preference pairs from {pairs_path}")

    # Split 90/10
    split_idx = int(len(records) * 0.9)
    train_ds = Dataset.from_list(records[:split_idx])
    eval_ds = Dataset.from_list(records[split_idx:])

    model, ref_model, tokenizer = load_model_and_tokenizer(args.model)

    out_dir = args.output_dir or "checkpoints/rlaif_dpo"

    training_args = DPOConfig(
        beta=0.1,                          # same as train_dpo.py
        loss_type="sigmoid",
        output_dir=out_dir,
        learning_rate=5e-5,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        max_grad_norm=1.0,
        bf16=True,
        gradient_checkpointing=True,
        dataloader_num_workers=4,
        remove_unused_columns=False,
        save_strategy="steps",
        save_steps=100,
        evaluation_strategy="steps",
        eval_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="rewards/margins",
        greater_is_better=True,
        push_to_hub=False,
        report_to="wandb",
        logging_steps=10,
        max_length=1024,
        max_prompt_length=512,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        callbacks=[LengthMonitorCallback()],
    )

    logger.info("Starting RLAIF-DPO training...")
    trainer.train()

    final_path = Path(out_dir) / "final"
    trainer.save_model(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    logger.info(f"RLAIF-DPO model saved to {final_path}")

    # Comparison table logged to W&B
    comparison = wandb.Table(
        columns=["Method", "Label source", "Scale", "Cost"],
        data=[
            ["RLHF",             "Humans",    "Limited (~1K)", "High"],
            ["RLAIF",            "AI judge",  "Unlimited",     "Low (API)"],
            ["RLVR",             "Symbolic",  "Unlimited",     "Zero (local)"],
            ["Constitutional AI","Self",       "Unlimited",     "Zero (local)"],
        ],
    )
    wandb.log({"alignment_method_comparison": comparison})


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:
    wandb.init(
        project="verifiable-alignment",
        name=f"rlaif-{args.stage}-{args.judge}",
        config=vars(args),
    )

    if args.stage == "label":
        label_stage(args)

    elif args.stage == "train":
        train_stage(args)

    elif args.stage == "both":
        pairs_path = label_stage(args)
        args.pairs = str(pairs_path)
        train_stage(args)

    else:
        raise ValueError(f"Unknown stage: {args.stage}")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--stage",
        choices=["label", "train", "both"],
        default="both",
        help="Pipeline stage to run.",
    )
    # Model
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Policy model for rollout generation and DPO training.",
    )
    # Judge
    parser.add_argument(
        "--judge",
        choices=["groq", "local"],
        default="groq",
        help="Judge backend.",
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
    # Labeling
    parser.add_argument(
        "--n_prompts",
        type=int,
        default=500,
        help="Number of MATH prompts to label.",
    )
    parser.add_argument(
        "--k_completions",
        type=int,
        default=4,
        help="Number of completions to generate per prompt.",
    )
    # Paths
    parser.add_argument(
        "--pairs",
        default="data/rlaif_pairs.jsonl",
        help="Path to save/load preference pairs.",
    )
    parser.add_argument(
        "--output",
        default="data/rlaif_pairs.jsonl",
        help="Output path for labeled pairs (used when --stage label).",
    )
    parser.add_argument(
        "--output_dir",
        default="checkpoints/rlaif_dpo",
        help="Output directory for DPO checkpoint.",
    )
    main(parser.parse_args())
