"""
data/generate_preferences.py

Self-play synthetic preference dataset generation.
- Loads GSM8K + MATH problems
- Uses Gemma-2-2B teacher to generate multiple candidate solutions
- Scores with symbolic math verifier (exact answer match)
- Constructs (chosen, rejected) pairs with difficulty stratification
- Publishes to HuggingFace Hub
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
from collections import defaultdict
from pathlib import Path

import torch
import wandb
from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset
from huggingface_hub import HfApi
from omegaconf import OmegaConf
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from data.verifier import MathVerifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
TEACHER_MODEL = "google/gemma-2-2b-it"
TEMPERATURES = [0.3, 0.7, 1.0]   # diversity sweep per problem
MAX_NEW_TOKENS = 512
TARGET_PAIRS = 10_000
DIFFICULTY_BUCKETS = {1: 0.15, 2: 0.20, 3: 0.25, 4: 0.25, 5: 0.15}  # MATH levels


# ── Prompt template ───────────────────────────────────────────────────────────
PROMPT_TEMPLATE = """Solve the following math problem step by step. \
Box your final answer with \\boxed{{}}.

Problem: {problem}

Solution:"""


# ── Data loading ──────────────────────────────────────────────────────────────
def load_problems() -> list[dict]:
    """Load and merge GSM8K + MATH problems."""
    gsm = load_dataset("gsm8k", "main", split="train")
    gsm_problems = [
        {"problem": ex["question"], "answer": ex["answer"].split("####")[-1].strip(),
         "source": "gsm8k", "difficulty": 2}  # GSM8K ≈ level 2
        for ex in gsm
    ]

    math = load_dataset("DigitalLearningGmbH/MATH-lighteval", split="train")
    math_problems = [
        {"problem": ex["problem"], "answer": ex["solution"],
         "source": "math", "difficulty": int(ex["level"].replace("Level ", "")) if ex["level"].replace("Level ", "").isdigit() else 3}
        for ex in math
    ]

    all_problems = gsm_problems + math_problems
    logger.info(f"Loaded {len(all_problems)} total problems "
                f"({len(gsm_problems)} GSM8K + {len(math_problems)} MATH)")
    return all_problems


# ── Candidate generation ──────────────────────────────────────────────────────
def generate_candidates(
    problems: list[dict],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    n_per_problem: int = 4,
    device: str = "cuda",
) -> list[dict]:
    """Generate n candidate solutions per problem across temperature sweep."""
    verifier = MathVerifier()
    results = []

    for problem in tqdm(problems, desc="Generating candidates"):
        prompt = PROMPT_TEMPLATE.format(problem=problem["problem"])
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        candidates = []
        for temp in TEMPERATURES:
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=temp,
                    top_p=0.95,
                    num_return_sequences=max(1, n_per_problem // len(TEMPERATURES)),
                    pad_token_id=tokenizer.eos_token_id,
                )
            decoded = tokenizer.batch_decode(
                outputs[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True
            )
            for text in decoded:
                is_correct = verifier.verify(text, problem["answer"])
                candidates.append({"text": text, "correct": is_correct, "temp": temp})

        results.append({**problem, "candidates": candidates})

    return results


# ── Pair construction ─────────────────────────────────────────────────────────
def build_preference_pairs(records: list[dict]) -> list[dict]:
    """
    Construct (chosen, rejected) pairs.
    chosen  = correct solution (randomly sampled from correct pool)
    rejected = incorrect solution (randomly sampled from incorrect pool)
    """
    import random

    pairs = []
    stats = defaultdict(int)

    for rec in records:
        correct = [c for c in rec["candidates"] if c["correct"]]
        incorrect = [c for c in rec["candidates"] if not c["correct"]]

        if not correct or not incorrect:
            stats["skipped_no_contrast"] += 1
            continue

        chosen = random.choice(correct)
        rejected = random.choice(incorrect)

        pairs.append({
            "prompt": PROMPT_TEMPLATE.format(problem=rec["problem"]),
            "chosen": chosen["text"],
            "rejected": rejected["text"],
            "difficulty": rec["difficulty"],
            "source": rec["source"],
            # dedup key
            "_hash": hashlib.md5(rec["problem"].encode()).hexdigest(),
        })
        stats["pairs_created"] += 1

    logger.info(f"Pair construction stats: {dict(stats)}")
    return pairs


# ── Difficulty stratification ─────────────────────────────────────────────────
def stratified_sample(pairs: list[dict], target: int = TARGET_PAIRS) -> list[dict]:
    """Sample pairs to match target difficulty distribution."""
    by_level: dict[int, list] = defaultdict(list)
    for p in pairs:
        by_level[p["difficulty"]].append(p)

    sampled = []
    for level, frac in DIFFICULTY_BUCKETS.items():
        n = int(target * frac)
        pool = by_level.get(level, [])
        sampled.extend(pool[:n] if len(pool) >= n else pool)
        logger.info(f"Level {level}: requested {n}, got {min(n, len(pool))}")

    return sampled


# ── Deduplication ─────────────────────────────────────────────────────────────
def deduplicate(pairs: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for p in pairs:
        if p["_hash"] not in seen:
            seen.add(p["_hash"])
            out.append({k: v for k, v in p.items() if k != "_hash"})
    logger.info(f"After dedup: {len(out)} pairs (removed {len(pairs) - len(out)})")
    return out


# ── HuggingFace Hub upload ────────────────────────────────────────────────────
def push_dataset(pairs: list[dict], hub_id: str) -> None:
    import random
    random.shuffle(pairs)
    split = int(len(pairs) * 0.95)
    train_data = Dataset.from_list(pairs[:split])
    val_data = Dataset.from_list(pairs[split:])
    ds = DatasetDict({"train": train_data, "validation": val_data})
    ds.push_to_hub(hub_id)
    logger.info(f"Dataset pushed to https://huggingface.co/datasets/{hub_id}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:
    wandb.init(project="verifiable-alignment", job_type="data-generation",
               config=vars(args))

    # Load problems
    problems = load_problems()

    # Optionally limit for debugging
    if args.limit:
        problems = problems[: args.limit]

    # Load teacher model
    logger.info(f"Loading teacher model: {TEACHER_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        TEACHER_MODEL, torch_dtype=torch.bfloat16, device_map="auto"
    )
    model.eval()

    # Generate candidates
    records = generate_candidates(problems, model, tokenizer)

    # Track rejection rate
    total_candidates = sum(len(r["candidates"]) for r in records)
    correct_candidates = sum(
        sum(1 for c in r["candidates"] if c["correct"]) for r in records
    )
    wandb.log({
        "data/total_candidates": total_candidates,
        "data/correct_rate": correct_candidates / total_candidates,
        "data/rejection_rate": 1 - correct_candidates / total_candidates,
    })

    # Build + filter pairs
    pairs = build_preference_pairs(records)
    pairs = stratified_sample(pairs)
    pairs = deduplicate(pairs)

    wandb.log({"data/final_pairs": len(pairs)})

    # Push to hub
    if args.push_to_hub:
        push_dataset(pairs, args.hub_id)

    # Save locally too
    out_path = Path("data/preferences.jsonl")
    import json
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
    logger.info(f"Saved {len(pairs)} pairs to {out_path}")
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/data_gen.yaml")
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument("--hub_id", default="SaiTejaSrivilli/verifiable-alignment-preferences")
    parser.add_argument("--limit", type=int, default=None, help="Limit problems (debug)")
    main(parser.parse_args())
