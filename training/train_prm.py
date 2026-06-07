"""
training/train_prm.py

Process Reward Model training on MATH-Shepherd step-level annotations.
- Lightweight linear head on top of frozen GRPO policy
- Scores each reasoning step independently
- Enables best-of-N reranking at inference via cumulative product (Lightman et al.)
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import wandb
from datasets import load_dataset
from omegaconf import OmegaConf
from sklearn.calibration import calibration_curve
from transformers import (
    AutoModel,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ── PRM model ─────────────────────────────────────────────────────────────────
class ProcessRewardModel(nn.Module):
    """
    Frozen base LM + trainable scalar head.
    Predicts P(step is correct | prefix so far).
    """

    def __init__(self, base_model_name: str, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.base = AutoModel.from_pretrained(
            base_model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )
        # Freeze the base
        for p in self.base.parameters():
            p.requires_grad_(False)

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        step_positions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns scalar reward for each step position.
        step_positions: [B, max_steps] — token indices of step end tokens
        """
        outputs = self.base(input_ids=input_ids, attention_mask=attention_mask)
        hidden = outputs.last_hidden_state  # [B, T, H]

        if step_positions is not None:
            # Gather hidden states at step boundary positions
            B, max_steps = step_positions.shape
            idx = step_positions.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
            step_hidden = hidden.gather(1, idx)  # [B, max_steps, H]
            logits = self.head(step_hidden).squeeze(-1)  # [B, max_steps]
        else:
            # Use last token hidden state
            logits = self.head(hidden[:, -1, :]).squeeze(-1)  # [B]

        return logits


# ── Dataset processing ────────────────────────────────────────────────────────
class MathShepherdDataset(torch.utils.data.Dataset):
    """
    Processes MATH-Shepherd step-level annotations.
    Each example has:
      - full solution text with step separators
      - per-step correctness labels (0/1)
    """

    STEP_SEP = "\n"

    def __init__(self, hf_dataset, tokenizer, max_length: int = 1024):
        self.data = hf_dataset
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ex = self.data[idx]
        full_text = ex.get("input", "") + " " + ex.get("label", "")

        enc = self.tokenizer(
            full_text,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        # Find step boundary positions (positions of \n tokens)
        newline_id = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.tokenize(self.STEP_SEP)
        )[-1]
        input_ids = enc["input_ids"].squeeze(0)
        step_positions = (input_ids == newline_id).nonzero(as_tuple=True)[0]

        # Pad step positions to fixed length
        max_steps = 16
        n_steps = min(len(step_positions), max_steps)
        padded_positions = torch.zeros(max_steps, dtype=torch.long)
        padded_positions[:n_steps] = step_positions[:n_steps]

        # Build step-level labels from annotation
        raw_labels = ex.get("label", "").strip()
        step_labels_str = [s.strip() for s in raw_labels.split(self.STEP_SEP) if s.strip()]
        step_labels = torch.zeros(max_steps)
        for i, lbl in enumerate(step_labels_str[:max_steps]):
            step_labels[i] = 1.0 if "+" in lbl else 0.0

        return {
            "input_ids": input_ids,
            "attention_mask": enc["attention_mask"].squeeze(0),
            "step_positions": padded_positions,
            "labels": step_labels,
            "n_steps": torch.tensor(n_steps),
        }


# ── Calibration callback ──────────────────────────────────────────────────────
class CalibrationCallback(TrainerCallback):
    """Plots reliability diagram to W&B after each epoch."""

    def __init__(self, eval_dataset, model, tokenizer, device: str = "cuda"):
        self.eval_dataset = eval_dataset
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def on_epoch_end(self, args, state, control, **kwargs):
        import numpy as np
        import matplotlib.pyplot as plt

        all_probs, all_labels = [], []

        self.model.eval()
        with torch.no_grad():
            for i in range(min(200, len(self.eval_dataset))):
                ex = self.eval_dataset[i]
                logits = self.model(
                    input_ids=ex["input_ids"].unsqueeze(0).to(self.device),
                    attention_mask=ex["attention_mask"].unsqueeze(0).to(self.device),
                    step_positions=ex["step_positions"].unsqueeze(0).to(self.device),
                )
                probs = torch.sigmoid(logits).cpu().numpy().flatten()
                labels = ex["labels"].numpy().flatten()
                n = ex["n_steps"].item()
                all_probs.extend(probs[:n].tolist())
                all_labels.extend(labels[:n].tolist())

        fraction_of_positives, mean_predicted_value = calibration_curve(
            all_labels, all_probs, n_bins=10
        )

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration")
        ax.plot(mean_predicted_value, fraction_of_positives, "b-o", label="PRM")
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Fraction of positives")
        ax.set_title(f"PRM Calibration — Epoch {state.epoch:.0f}")
        ax.legend()
        wandb.log({"eval/prm_calibration": wandb.Image(fig), "epoch": state.epoch})
        plt.close(fig)


# ── Custom Trainer for PRM ────────────────────────────────────────────────────
class PRMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        n_steps = inputs.pop("n_steps")

        logits = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            step_positions=inputs["step_positions"],
        )  # [B, max_steps]

        # Mask padding steps
        B, max_steps = logits.shape
        mask = torch.arange(max_steps, device=logits.device).unsqueeze(0) < n_steps.unsqueeze(1)

        loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
        loss = (loss * mask.float()).sum() / mask.float().sum()

        return (loss, logits) if return_outputs else loss


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:
    cfg = OmegaConf.load(args.config)

    wandb.init(
        project=cfg.logging.project,
        name=cfg.logging.run_name,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = ProcessRewardModel(
        base_model_name=cfg.model.name,
        hidden_size=cfg.prm_head.hidden_size,
        dropout=cfg.prm_head.dropout,
    )

    raw_ds = load_dataset(cfg.data.dataset_id)
    train_dataset = MathShepherdDataset(
        raw_ds[cfg.data.train_split], tokenizer, cfg.data.max_length
    )
    eval_dataset = MathShepherdDataset(
        raw_ds[cfg.data.eval_split], tokenizer, cfg.data.max_length
    )

    training_args = TrainingArguments(
        output_dir=cfg.output.output_dir,
        learning_rate=cfg.training.learning_rate,
        lr_scheduler_type=cfg.training.lr_scheduler_type,
        warmup_ratio=cfg.training.warmup_ratio,
        per_device_train_batch_size=cfg.training.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.training.gradient_accumulation_steps,
        num_train_epochs=cfg.training.num_train_epochs,
        max_grad_norm=cfg.training.max_grad_norm,
        bf16=cfg.training.bf16,
        gradient_checkpointing=cfg.training.gradient_checkpointing,
        save_strategy=cfg.training.save_strategy,
        evaluation_strategy=cfg.training.evaluation_strategy,
        load_best_model_at_end=cfg.training.load_best_model_at_end,
        metric_for_best_model=cfg.training.metric_for_best_model,
        push_to_hub=cfg.training.push_to_hub,
        hub_model_id=cfg.training.hub_model_id,
        report_to=cfg.logging.report_to,
        logging_steps=cfg.logging.logging_steps,
    )

    calibration_cb = CalibrationCallback(
        eval_dataset=eval_dataset, model=model, tokenizer=tokenizer
    )

    trainer = PRMTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[calibration_cb],
    )

    logger.info("Starting PRM training...")
    trainer.train()

    out_path = Path(cfg.output.output_dir) / "final"
    trainer.save_model(str(out_path))
    logger.info(f"PRM saved to {out_path}")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/prm_base.yaml")
    main(parser.parse_args())
