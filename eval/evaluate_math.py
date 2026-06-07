"""
eval/evaluate_math.py

Comprehensive evaluation using EleutherAI lm-evaluation-harness.
- MATH accuracy by difficulty level (1-5)
- GSM8K 8-shot chain-of-thought
- Best-of-N with PRM reranking
- All results logged to W&B
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
import wandb
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Model registry ─────────────────────────────────────────────────────────────
MODEL_IDS = {
    "base":     "mistralai/Mistral-7B-v0.3",
    "dpo":      "SaiTejaSrivilli/verifiable-alignment-dpo",
    "grpo":     "SaiTejaSrivilli/verifiable-alignment-grpo",
    "grpo_prm": "SaiTejaSrivilli/verifiable-alignment-grpo",   # + PRM reranking
}

PRM_MODEL_ID = "SaiTejaSrivilli/verifiable-alignment-prm"

# ── lm-eval task config ───────────────────────────────────────────────────────
TASKS = ["minerva_math", "gsm8k"]
NUM_FEWSHOT = {"minerva_math": 4, "gsm8k": 8}


# ── Standard evaluation via lm-eval ──────────────────────────────────────────
def run_lm_eval(model_id: str, tasks: list[str], limit: Optional[int] = None) -> dict:
    """
    Run lm-evaluation-harness on the given tasks.
    Returns {task: {metric: value}} dict.
    """
    lm = HFLM(
        pretrained=model_id,
        dtype="bfloat16",
        device="cuda",
        batch_size=8,
    )

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=tasks,
        num_fewshot=None,      # lm-eval uses task-level defaults
        limit=limit,
        log_samples=True,
    )
    return results["results"]


# ── MATH difficulty breakdown ─────────────────────────────────────────────────
def math_by_difficulty(model_id: str, limit: Optional[int] = None) -> dict[int, float]:
    """
    Evaluate MATH accuracy broken down by difficulty level 1-5.
    Uses MATH subtasks in lm-eval (math_algebra, math_counting_and_probability, etc.)
    Approximates level breakdown via problem metadata.
    """
    from datasets import load_dataset
    from transformers import pipeline

    math_ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="test")
    if limit:
        math_ds = math_ds.select(range(limit))

    gen = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    from data.verifier import MathVerifier
    verifier = MathVerifier()

    level_correct: dict[int, int] = {i: 0 for i in range(1, 6)}
    level_total: dict[int, int] = {i: 0 for i in range(1, 6)}

    for ex in math_ds:
        level = int(ex["level"].replace("Level ", ""))
        prompt = (
            "Solve the following math problem step by step. "
            "Box your final answer with \\boxed{}.\n\n"
            f"Problem: {ex['problem']}\n\nSolution:"
        )
        output = gen(prompt, max_new_tokens=512, do_sample=False)[0]["generated_text"]
        completion = output[len(prompt):]

        level_total[level] += 1
        if verifier.verify(completion, ex["solution"]):
            level_correct[level] += 1

    return {lvl: level_correct[lvl] / max(1, level_total[lvl]) for lvl in range(1, 6)}


# ── Best-of-N with PRM ────────────────────────────────────────────────────────
def evaluate_bon_prm(
    policy_model_id: str,
    prm_model_id: str,
    n: int = 8,
    limit: int = 200,
) -> float:
    """
    Best-of-N reranking using PRM cumulative product (Lightman et al. 2023).
    Returns accuracy on MATH test set.
    """
    from datasets import load_dataset
    from data.verifier import MathVerifier

    # Lazy import to avoid loading two models unless needed
    from training.train_prm import ProcessRewardModel

    verifier = MathVerifier()
    math_ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="test").select(range(limit))

    # Load policy
    tokenizer = AutoTokenizer.from_pretrained(policy_model_id)
    policy = AutoModelForCausalLM.from_pretrained(
        policy_model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    # Load PRM
    prm = ProcessRewardModel(
        base_model_name=policy_model_id,
        hidden_size=4096,
    )
    prm_state = torch.load(f"{prm_model_id}/pytorch_model.bin", map_location="cpu")
    prm.load_state_dict(prm_state)
    prm.eval().to("cuda")

    correct = 0

    for ex in math_ds:
        prompt = (
            "Solve the following math problem step by step. "
            "Box your final answer with \\boxed{}.\n\n"
            f"Problem: {ex['problem']}\n\nSolution:"
        )
        enc = tokenizer(prompt, return_tensors="pt").to("cuda")

        # Generate N candidates
        with torch.no_grad():
            outputs = policy.generate(
                **enc,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.9,
                num_return_sequences=n,
                pad_token_id=tokenizer.eos_token_id,
            )
        candidates = tokenizer.batch_decode(
            outputs[:, enc["input_ids"].shape[1]:], skip_special_tokens=True
        )

        # Score each candidate with PRM (cumulative product of step probs)
        best_score = -float("inf")
        best_candidate = candidates[0]

        for cand in candidates:
            full_text = prompt + cand
            prm_enc = tokenizer(
                full_text, return_tensors="pt", max_length=1024, truncation=True
            ).to("cuda")
            with torch.no_grad():
                logits = prm(**prm_enc)
            step_probs = torch.sigmoid(logits).cpu().squeeze()
            # Cumulative product (Lightman et al.) — log-sum for numerical stability
            score = step_probs.log().sum().item()
            if score > best_score:
                best_score = score
                best_candidate = cand

        if verifier.verify(best_candidate, ex["solution"]):
            correct += 1

    return correct / limit


# ── Results table ─────────────────────────────────────────────────────────────
def build_results_table(all_results: dict) -> pd.DataFrame:
    rows = []
    for model_name, res in all_results.items():
        row = {"model": model_name}
        if "math_accuracy_overall" in res:
            row["MATH Overall"] = f"{res['math_accuracy_overall'] * 100:.1f}%"
        for lvl in range(1, 6):
            key = f"math_level_{lvl}"
            if key in res:
                row[f"MATH L{lvl}"] = f"{res[key] * 100:.1f}%"
        if "gsm8k_accuracy" in res:
            row["GSM8K"] = f"{res['gsm8k_accuracy'] * 100:.1f}%"
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:
    wandb.init(project="verifiable-alignment", job_type="evaluation", name="eval-all")

    models_to_eval = args.models or list(MODEL_IDS.keys())
    all_results = {}

    for model_name in models_to_eval:
        model_id = MODEL_IDS[model_name]
        logger.info(f"\n{'='*60}\nEvaluating: {model_name} ({model_id})\n{'='*60}")

        res = {}

        # Standard lm-eval tasks
        logger.info("Running lm-evaluation-harness...")
        lm_results = run_lm_eval(model_id, TASKS, limit=args.limit)

        if "minerva_math" in lm_results:
            res["math_accuracy_overall"] = lm_results["minerva_math"].get("exact_match,none", 0)
        if "gsm8k" in lm_results:
            res["gsm8k_accuracy"] = lm_results["gsm8k"].get("exact_match,strict-match", 0)

        # MATH by difficulty level
        logger.info("Running MATH difficulty breakdown...")
        level_accs = math_by_difficulty(model_id, limit=args.limit)
        for lvl, acc in level_accs.items():
            res[f"math_level_{lvl}"] = acc

        # Best-of-N with PRM (only for grpo_prm)
        if model_name == "grpo_prm":
            logger.info("Running Best-of-8 PRM reranking...")
            bon_acc = evaluate_bon_prm(model_id, PRM_MODEL_ID, n=8, limit=args.limit or 200)
            res["math_bon8_prm"] = bon_acc

        all_results[model_name] = res

        # Log to W&B
        wandb.log({f"{model_name}/{k}": v for k, v in res.items()})

    # Build and log results table
    df = build_results_table(all_results)
    logger.info(f"\n{df.to_string()}")
    wandb.log({"results_table": wandb.Table(dataframe=df.reset_index())})

    # Save to disk
    out_path = Path("eval/results.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved to {out_path}")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--models", nargs="+", choices=list(MODEL_IDS.keys()),
        default=None, help="Models to evaluate (default: all)"
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit eval examples (debug)")
    main(parser.parse_args())
