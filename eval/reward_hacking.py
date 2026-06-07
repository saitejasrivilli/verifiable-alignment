"""
eval/reward_hacking.py

Reward hacking detection and analysis.
- Computes Pearson correlation between RM score and symbolic verifier
- Identifies cases where RM is high but answer is wrong (reward hacking)
- Logs top-20 hacking examples to W&B Table for inspection
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import wandb
from datasets import load_dataset
from scipy.stats import pearsonr
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline

from data.verifier import MathVerifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── Reward scoring ─────────────────────────────────────────────────────────────
def score_with_reward_model(
    model_id: str,
    prompts: list[str],
    completions: list[str],
    batch_size: int = 8,
) -> list[float]:
    """
    Score completions using a reward model (DPO implicit reward).
    Uses the log-ratio: log π_θ(y|x) / π_ref(y|x) as reward proxy.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    policy = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )
    ref_model_id = "mistralai/Mistral-7B-v0.3"
    ref = AutoModelForCausalLM.from_pretrained(
        ref_model_id, torch_dtype=torch.bfloat16, device_map="auto"
    )

    scores = []
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        batch_completions = completions[i:i + batch_size]
        full_texts = [p + c for p, c in zip(batch_prompts, batch_completions)]

        enc = tokenizer(
            full_texts, return_tensors="pt", padding=True, truncation=True, max_length=1024
        ).to("cuda")

        with torch.no_grad():
            policy_logits = policy(**enc).logits
            ref_logits = ref(**enc).logits

        # Log-ratio (DPO implicit reward)
        policy_lp = torch.log_softmax(policy_logits, dim=-1)
        ref_lp = torch.log_softmax(ref_logits, dim=-1)

        labels = enc["input_ids"][:, 1:]
        policy_nll = -policy_lp[:, :-1].gather(2, labels.unsqueeze(-1)).squeeze(-1)
        ref_nll = -ref_lp[:, :-1].gather(2, labels.unsqueeze(-1)).squeeze(-1)

        mask = enc["attention_mask"][:, 1:].float()
        log_ratio = ((ref_nll - policy_nll) * mask).sum(-1) / mask.sum(-1)
        scores.extend(log_ratio.cpu().tolist())

    return scores


# ── Reward hacking index ───────────────────────────────────────────────────────
def compute_reward_hacking_index(
    model_id: str,
    n_samples: int = 1000,
) -> dict:
    """
    Reward hacking index = 1 - Pearson(RM_score, symbolic_correct).
    High value → reward model is poorly calibrated.

    Also identifies top-K hacking examples:
      high RM score but wrong answer (over-reward)
      low RM score but correct answer (under-reward)
    """
    verifier = MathVerifier()
    math_ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="test").select(range(n_samples))

    gen = pipeline(
        "text-generation",
        model=model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        pad_token_id=AutoTokenizer.from_pretrained(model_id).eos_token_id,
    )

    prompts, completions, gt_labels = [], [], []

    for ex in math_ds:
        prompt = (
            "Solve the following math problem step by step. "
            "Box your final answer with \\boxed{}.\n\n"
            f"Problem: {ex['problem']}\n\nSolution:"
        )
        output = gen(prompt, max_new_tokens=512, do_sample=False)[0]["generated_text"]
        completion = output[len(prompt):]

        prompts.append(prompt)
        completions.append(completion)
        gt_labels.append(1 if verifier.verify(completion, ex["solution"]) else 0)

    rm_scores = score_with_reward_model(model_id, prompts, completions)

    # Pearson correlation
    corr, p_value = pearsonr(rm_scores, gt_labels)
    hacking_index = 1 - corr

    logger.info(f"Reward hacking index: {hacking_index:.4f} (Pearson r={corr:.4f}, p={p_value:.4e})")

    # Find top hacking examples (high RM score, wrong answer)
    hacking_examples = []
    for i, (score, label, prompt, completion) in enumerate(
        zip(rm_scores, gt_labels, prompts, completions)
    ):
        if score > np.percentile(rm_scores, 75) and label == 0:
            hacking_examples.append({
                "problem": prompt.split("Problem:")[-1].split("\n\nSolution:")[0].strip(),
                "completion": completion[:300],
                "rm_score": round(score, 4),
                "correct": bool(label),
            })

    hacking_examples.sort(key=lambda x: x["rm_score"], reverse=True)

    return {
        "reward_hacking_index": hacking_index,
        "pearson_r": corr,
        "p_value": p_value,
        "accuracy": sum(gt_labels) / len(gt_labels),
        "top_hacking_examples": hacking_examples[:20],
    }


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:
    wandb.init(project="verifiable-alignment", job_type="reward-hacking-analysis")

    model_ids = {
        "dpo": "SaiTejaSrivilli/verifiable-alignment-dpo",
        "grpo": "SaiTejaSrivilli/verifiable-alignment-grpo",
    }

    for model_name, model_id in model_ids.items():
        logger.info(f"Analysing reward hacking for: {model_name}")
        results = compute_reward_hacking_index(model_id, n_samples=args.n_samples)

        wandb.log({
            f"{model_name}/reward_hacking_index": results["reward_hacking_index"],
            f"{model_name}/pearson_r": results["pearson_r"],
            f"{model_name}/accuracy": results["accuracy"],
        })

        # Log hacking examples as a W&B Table
        table = wandb.Table(columns=["problem", "completion", "rm_score", "correct"])
        for ex in results["top_hacking_examples"]:
            table.add_data(ex["problem"], ex["completion"], ex["rm_score"], ex["correct"])
        wandb.log({f"{model_name}/reward_hacking_examples": table})

        logger.info(
            f"{model_name} — hacking index: {results['reward_hacking_index']:.4f}, "
            f"accuracy: {results['accuracy'] * 100:.1f}%"
        )

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_samples", type=int, default=1000)
    main(parser.parse_args())
