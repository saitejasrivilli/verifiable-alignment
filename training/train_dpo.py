"""
training/train_dpo.py

DPO fine-tuning with TRL DPOTrainer + LoRA.
Tracks: reward margin, KL divergence, MATH accuracy.
"""

from __future__ import annotations

import argparse
import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import wandb
from datasets import load_dataset
from omegaconf import OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainerCallback,
)
from trl import DPOConfig, DPOTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── KL tracking callback ──────────────────────────────────────────────────────
class KLDivergenceCallback(TrainerCallback):
    """
    Computes KL divergence between policy and reference model on a fixed
    500-sample reference set every eval_steps. Logged to W&B.
    """

    def __init__(self, ref_model, tokenizer, reference_texts: list[str], device: str):
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.reference_texts = reference_texts[:500]
        self.device = device

    @torch.no_grad()
    def on_evaluate(self, args, state, control, model=None, **kwargs):
        if model is None:
            return

        kl_values = []
        for text in self.reference_texts[:50]:  # subsample for speed
            enc = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512
            ).to(self.device)

            policy_logits = model(**enc).logits
            ref_logits = self.ref_model(**enc).logits

            policy_lp = torch.log_softmax(policy_logits, dim=-1)
            ref_lp = torch.log_softmax(ref_logits, dim=-1)

            kl = (policy_lp.exp() * (policy_lp - ref_lp)).sum(-1).mean().item()
            kl_values.append(kl)

        mean_kl = sum(kl_values) / len(kl_values)
        wandb.log({"eval/kl_divergence": mean_kl, "train/global_step": state.global_step})
        logger.info(f"Step {state.global_step} — KL divergence: {mean_kl:.4f}")


# ── Length distribution callback ──────────────────────────────────────────────
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
                    f"⚠️  Length collapse detected at step {state.global_step}: "
                    f"{length:.0f} vs baseline {self.baseline_length:.0f}"
                )
                wandb.alert(
                    title="Length Collapse",
                    text=f"Mean chosen length dropped to {length:.0f} "
                         f"({100 * length / self.baseline_length:.0f}% of baseline)",
                    level=wandb.AlertLevel.WARN,
                )


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model_and_tokenizer(cfg):
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"   # DPO requires left-padding

    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        torch_dtype=getattr(torch, cfg.model.torch_dtype),
        attn_implementation=cfg.model.attn_implementation,
        device_map="auto",
    )

    # Reference model — separate copy, frozen
    ref_model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name,
        torch_dtype=getattr(torch, cfg.model.torch_dtype),
        attn_implementation=cfg.model.attn_implementation,
        device_map="auto",
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    # Apply LoRA to policy model
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

    return model, ref_model, tokenizer


# ── Dataset loading ───────────────────────────────────────────────────────────
def load_data(cfg):
    ds = load_dataset(cfg.data.dataset_id)
    train_ds = ds[cfg.data.train_split]
    eval_ds = ds[cfg.data.eval_split]
    logger.info(f"Train: {len(train_ds)} | Eval: {len(eval_ds)}")
    return train_ds, eval_ds


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:
    cfg = OmegaConf.load(args.config)

    # Override cfg from CLI (for sweep)
    if args.beta is not None:
        cfg.training.beta = args.beta
    if args.lr is not None:
        cfg.training.learning_rate = args.lr

    wandb.init(
        project=cfg.logging.project,
        name=f"{cfg.logging.run_name}-beta{cfg.training.beta}",
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    model, ref_model, tokenizer = load_model_and_tokenizer(cfg)
    train_ds, eval_ds = load_data(cfg)

    # Build DPOConfig
    training_args = DPOConfig(
        beta=cfg.training.beta,
        loss_type=cfg.training.loss_type,
        output_dir=cfg.output.output_dir,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        warmup_ratio=cfg.training.warmup_ratio,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.training.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        num_train_epochs=cfg.training.num_train_epochs,
        max_grad_norm=cfg.training.max_grad_norm,
        weight_decay=cfg.training.weight_decay,
        bf16=cfg.training.bf16,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        dataloader_num_workers=cfg.training.dataloader_num_workers,
        remove_unused_columns=cfg.training.remove_unused_columns,
        save_strategy=cfg.training.save_strategy,
        save_steps=cfg.training.save_steps,
        evaluation_strategy=cfg.training.evaluation_strategy,
        eval_steps=cfg.training.eval_steps,
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        metric_for_best_model=cfg.training.metric_for_best_model,
        greater_is_better=cfg.training.greater_is_better,
        push_to_hub=cfg.training.push_to_hub,
        hub_model_id=cfg.training.hub_model_id,
        hub_strategy=cfg.training.hub_strategy,
        report_to=cfg.logging.report_to,
        logging_steps=cfg.logging.logging_steps,
        max_length=cfg.data.max_length,
        max_prompt_length=cfg.data.max_prompt_length,
    )

    # Build callbacks
    kl_callback = KLDivergenceCallback(
        ref_model=ref_model,
        tokenizer=tokenizer,
        reference_texts=[ex["prompt"] + ex["chosen"] for ex in eval_ds.select(range(500))],
        device="cuda",
    )
    length_callback = LengthMonitorCallback()

    trainer = DPOTrainer(
        model=model,
        ref_model=ref_model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        callbacks=[kl_callback, length_callback],
    )

    logger.info("Starting DPO training...")
    trainer.train()

    # Save final model
    out_path = Path(cfg.output.output_dir) / "final"
    trainer.save_model(str(out_path))
    tokenizer.save_pretrained(str(out_path))
    logger.info(f"Model saved to {out_path}")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dpo_base.yaml")
    parser.add_argument("--beta", type=float, default=None, help="Override beta (for sweep)")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    main(parser.parse_args())
