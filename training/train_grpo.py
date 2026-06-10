"""
training/train_grpo.py

Group Relative Policy Optimization with verifiable math rewards.
- G=8 rollouts per problem
- Symbolic verifier as reward function (no reward model needed)
- Correct advantage computation: group-normalize THEN clip
- Entropy bonus + length regularisation
- DeepSpeed ZeRO-2 via accelerate/deepspeed launcher
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import torch
import wandb
from datasets import load_dataset
from omegaconf import OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOConfig, GRPOTrainer

from data.verifier import MathVerifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

VERIFIER = MathVerifier()


# ── Reward function ───────────────────────────────────────────────────────────
def verifiable_reward(
    prompts: list[str],
    completions: list[str],
    ground_truths: list[str],
    length_penalty_coeff: float = 0.01,
) -> list[float]:
    """
    Reward function for GRPO.

    Base reward:
      +1.0  if symbolic verifier confirms correct answer
      -1.0  if answer is wrong
      -0.5  if no answer extracted (format failure)

    Length regularisation:
      Small penalty for very short responses to prevent degenerate collapse.
    """
    rewards = []
    for completion, truth in zip(completions, ground_truths):
        extracted = VERIFIER.extract_answer(completion)

        if extracted is None:
            base = -0.5      # format failure
        elif VERIFIER.verify(completion, truth):
            base = 1.0       # correct
        else:
            base = -1.0      # wrong answer

        # Length regularisation: penalise responses shorter than 50 tokens
        token_count = len(completion.split())
        length_pen = length_penalty_coeff * max(0, 50 - token_count)

        rewards.append(base - length_pen)

    return rewards


# Advantage normalisation is handled internally by TRL's GRPOTrainer
# (group-relative normalisation with configurable clip range, controlled
# via the `epsilon` parameter in GRPOConfig).
# For the DAPO variant with asymmetric clipping and dynamic group skipping,
# see training/train_dapo.py which implements a custom DAPOTrainer.


# ── Clip ratio monitoring callback ────────────────────────────────────────────
class ClipFractionCallback(TrainerCallback):
    """
    Logs clip ratio to W&B.
    TRL's GRPOTrainer logs this metric as `train/clip_ratio`.
    If clip_ratio > 0.3 consistently → LR is too high.
    """

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and "train/clip_ratio" in logs:
            cf = logs["train/clip_ratio"]
            if cf > 0.3:
                logger.warning(
                    f"High clip ratio at step {state.global_step}: {cf:.3f} "
                    "(consider reducing learning rate)"
                )


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model_and_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        torch_dtype=getattr(torch, cfg.model.torch_dtype),
        attn_implementation=cfg.model.attn_implementation,
        device_map="auto",
    )

    lora_cfg = LoraConfig(
        r=cfg.lora.r,
        lora_alpha=cfg.lora.lora_alpha,
        lora_dropout=cfg.lora.lora_dropout,
        target_modules=list(cfg.lora.target_modules),
        bias=cfg.lora.bias,
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer


# ── Dataset ───────────────────────────────────────────────────────────────────
def load_data(cfg):
    ds = load_dataset(cfg.data.dataset_id)
    train_ds = ds[cfg.data.train_split]
    eval_ds = ds[cfg.data.eval_split]

    def format_example(ex):
        return {
            "prompt": f"Solve the following math problem step by step. "
                      f"Box your final answer with \\boxed{{}}.\n\n"
                      f"Problem: {ex[cfg.data.problem_field]}\n\nSolution:",
            "ground_truth": ex[cfg.data.solution_field],
        }

    train_ds = train_ds.map(format_example)
    eval_ds = eval_ds.map(format_example)
    return train_ds, eval_ds


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:
    cfg = OmegaConf.load(args.config)

    wandb.init(
        project=cfg.logging.project,
        name=cfg.logging.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    model, tokenizer = load_model_and_tokenizer(cfg)
    train_ds, eval_ds = load_data(cfg)

    # Reward function closure capturing ground truths
    def reward_fn(prompts, completions, **kwargs):
        ground_truths = kwargs.get("ground_truth", [""] * len(completions))
        return verifiable_reward(
            prompts=prompts,
            completions=completions,
            ground_truths=ground_truths,
            length_penalty_coeff=cfg.grpo.length_penalty_coeff,
        )

    training_args = GRPOConfig(
        output_dir=cfg.output.output_dir,
        num_generations=cfg.grpo.num_generations,
        temperature=cfg.grpo.temperature,
        top_p=cfg.grpo.top_p,
        max_new_tokens=cfg.data.max_new_tokens,
        max_prompt_length=cfg.data.max_prompt_length,
        # GRPO-specific
        epsilon=cfg.grpo.clip_range,
        beta=cfg.grpo.kl_coeff,
        # Training
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        warmup_ratio=cfg.training.warmup_ratio,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        num_train_epochs=cfg.training.num_train_epochs,
        max_grad_norm=cfg.training.max_grad_norm,
        bf16=cfg.training.bf16,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        # vLLM for fast rollouts
        use_vllm=cfg.training.use_vllm,
        vllm_gpu_memory_utilization=cfg.training.vllm_gpu_memory_utilization,
        # Checkpointing
        save_strategy=cfg.training.save_strategy,
        save_steps=cfg.training.save_steps,
        evaluation_strategy=cfg.training.evaluation_strategy,
        eval_steps=cfg.training.eval_steps,
        push_to_hub=cfg.training.push_to_hub,
        hub_model_id=cfg.training.hub_model_id,
        hub_strategy=cfg.training.hub_strategy,
        # Logging
        report_to=cfg.logging.report_to,
        logging_steps=cfg.logging.logging_steps,
        # DeepSpeed
        deepspeed=args.deepspeed,
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        reward_funcs=reward_fn,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        callbacks=[ClipFractionCallback()],
    )

    logger.info("Starting GRPO training...")
    trainer.train()

    out_path = Path(cfg.output.output_dir) / "final"
    trainer.save_model(str(out_path))
    tokenizer.save_pretrained(str(out_path))
    logger.info(f"GRPO model saved to {out_path}")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/grpo_base.yaml")
    parser.add_argument("--deepspeed", default="configs/ds_zero2.json")
    main(parser.parse_args())
