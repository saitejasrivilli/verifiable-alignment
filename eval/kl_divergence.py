"""
eval/kl_divergence.py

KL divergence analysis between trained policy and base reference.
- Evaluated on a fixed 500-sample reference set (reproducible)
- Logged per training stage (DPO, GRPO)
- Generates KL-vs-accuracy tradeoff plot
"""

from __future__ import annotations

import argparse
import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REFERENCE_MODEL = "mistralai/Mistral-7B-v0.3"
REFERENCE_SEED = 42     # fixed seed for reproducible reference set
N_REFERENCE = 500


def load_reference_set(n: int = N_REFERENCE) -> list[str]:
    """Fixed 500-sample reference set — same across all eval runs."""
    ds = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="test").shuffle(seed=REFERENCE_SEED).select(range(n))
    return [
        f"Solve the following math problem step by step. "
        f"Box your final answer with \\boxed{{}}.\n\nProblem: {ex['problem']}\n\nSolution:"
        for ex in ds
    ]


@torch.no_grad()
def compute_kl(
    policy_model: AutoModelForCausalLM,
    ref_model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    texts: list[str],
    batch_size: int = 4,
    device: str = "cuda",
) -> float:
    """
    KL(π_θ || π_ref) averaged over the reference text set.
    """
    kl_values = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        policy_logits = policy_model(**enc).logits
        ref_logits = ref_model(**enc).logits

        policy_lp = torch.log_softmax(policy_logits.float(), dim=-1)
        ref_lp = torch.log_softmax(ref_logits.float(), dim=-1)

        # Token-level KL, averaged over sequence and batch
        kl = (policy_lp.exp() * (policy_lp - ref_lp)).sum(-1)  # [B, T]
        mask = enc["attention_mask"].float()
        kl_mean = (kl * mask).sum() / mask.sum()
        kl_values.append(kl_mean.item())

    return float(np.mean(kl_values))


def main(args: argparse.Namespace) -> None:
    wandb.init(project="verifiable-alignment", job_type="kl-analysis")

    reference_texts = load_reference_set()

    model_configs = {
        "base":  ("mistralai/Mistral-7B-v0.3", 0.0),     # KL=0 by definition
        "dpo":   ("SaiTejaSrivilli/verifiable-alignment-dpo", None),
        "grpo":  ("SaiTejaSrivilli/verifiable-alignment-grpo", None),
    }

    # From eval results (update with actual numbers)
    accuracy_map = {
        "base":  0.124,
        "dpo":   0.197,
        "grpo":  0.263,
    }

    tokenizer = AutoTokenizer.from_pretrained(REFERENCE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ref_model = AutoModelForCausalLM.from_pretrained(
        REFERENCE_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    ref_model.eval()

    kl_results = {"base": 0.0}

    for model_name, (model_id, preset_kl) in model_configs.items():
        if preset_kl is not None:
            kl_results[model_name] = preset_kl
            continue

        logger.info(f"Computing KL for {model_name}...")
        policy = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto"
        )
        policy.eval()

        kl = compute_kl(policy, ref_model, tokenizer, reference_texts)
        kl_results[model_name] = kl
        logger.info(f"{model_name} — KL divergence: {kl:.4f}")
        wandb.log({f"{model_name}/kl_divergence": kl})

        del policy
        torch.cuda.empty_cache()

    # KL vs accuracy tradeoff plot
    fig, ax = plt.subplots(figsize=(7, 5))
    names = list(kl_results.keys())
    kls = [kl_results[n] for n in names]
    accs = [accuracy_map.get(n, 0) * 100 for n in names]

    ax.scatter(kls, accs, s=120, zorder=5)
    for name, kl, acc in zip(names, kls, accs):
        ax.annotate(name, (kl, acc), textcoords="offset points", xytext=(8, 4), fontsize=11)

    ax.plot(kls, accs, "k--", alpha=0.3)
    ax.set_xlabel("KL Divergence from Base Model", fontsize=12)
    ax.set_ylabel("MATH Accuracy (%)", fontsize=12)
    ax.set_title("KL–Accuracy Tradeoff Across Training Stages", fontsize=13)
    ax.grid(True, alpha=0.3)

    wandb.log({"kl_accuracy_tradeoff": wandb.Image(fig)})
    plt.savefig("eval/kl_accuracy_tradeoff.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved KL–accuracy plot.")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    main(parser.parse_args())
