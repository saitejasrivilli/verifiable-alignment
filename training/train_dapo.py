"""
training/train_dapo.py

DAPO: Direct Advantage Policy Optimization
Improvements over GRPO for long-horizon mathematical reasoning:
  1. Clip-higher: asymmetric PPO clip (low=0.2, high=0.28)
  2. Token-level policy gradient instead of sequence-level
  3. Dynamic sampling: skip zero-advantage groups
  4. Entropy bonus to prevent mode collapse
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import wandb
from datasets import load_dataset
from omegaconf import OmegaConf
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
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
    rewards = []
    for completion, truth in zip(completions, ground_truths):
        extracted = VERIFIER.extract_answer(completion)

        if extracted is None:
            base = -0.5
        elif VERIFIER.verify(completion, truth):
            base = 1.0
        else:
            base = -1.0

        token_count = len(completion.split())
        length_pen = length_penalty_coeff * max(0, 50 - token_count)

        rewards.append(base - length_pen)

    return rewards


# ── Advantage computation ─────────────────────────────────────────────────────
def compute_group_advantages(
    rewards: torch.Tensor,
    clip_range: float = 5.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Group-relative advantage normalisation.
    Returns (advantages, is_zero_advantage).

    is_zero_advantage is True when all rewards in the group are identical
    (zero std → no learning signal, DAPO skips these groups).
    """
    std = rewards.std()
    if std < eps:
        return torch.zeros_like(rewards), True
    advantages = (rewards - rewards.mean()) / (std + eps)
    advantages = torch.clamp(advantages, -clip_range, clip_range)
    return advantages, False


# ── DAPO policy loss ──────────────────────────────────────────────────────────
def dapo_policy_loss(
    log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    attention_mask: torch.Tensor,
    clip_low: float = 0.2,
    clip_high: float = 0.28,
) -> torch.Tensor:
    """
    Token-level DAPO policy gradient loss with asymmetric clipping.

    log_probs, old_log_probs: [B, T] — per-token log probabilities
    advantages: [B] — per-sequence advantage (broadcast to token level)
    attention_mask: [B, T]
    clip_low / clip_high: asymmetric PPO epsilon values

    The ratio is clipped to [1-clip_low, 1+clip_high] instead of the
    symmetric [1-ε, 1+ε] used in standard GRPO/PPO.
    """
    # Per-token probability ratio
    log_ratio = log_probs - old_log_probs
    ratio = torch.exp(log_ratio)

    # Broadcast advantage [B] → [B, T]
    adv = advantages.unsqueeze(1).expand_as(log_probs)

    # Asymmetric clipping
    ratio_clipped = torch.clamp(ratio, 1.0 - clip_low, 1.0 + clip_high)

    # PPO objective: min(unclipped, clipped) surrogate
    surr1 = ratio * adv
    surr2 = ratio_clipped * adv
    per_token_loss = -torch.min(surr1, surr2)

    # Mask padding, average over non-padding tokens
    masked_loss = per_token_loss * attention_mask
    loss = masked_loss.sum() / attention_mask.sum().clamp(min=1)
    return loss


# ── Entropy bonus ─────────────────────────────────────────────────────────────
def compute_entropy(
    logits: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean per-token entropy over non-padded positions."""
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    entropy = -(probs * log_probs).sum(dim=-1)   # [B, T]
    masked = entropy * attention_mask
    return masked.sum() / attention_mask.sum().clamp(min=1)


# ── DAPO training step ────────────────────────────────────────────────────────
class DAPOTrainer:
    """
    Wraps a causal LM and implements the DAPO training loop manually.

    DAPO is not natively supported by TRL, so we implement the key loop
    directly: rollout → filter zero-advantage groups → token-level loss
    → entropy bonus.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        ref_model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        cfg,
        args: argparse.Namespace,
    ):
        self.model = model
        self.ref_model = ref_model
        self.tokenizer = tokenizer
        self.cfg = cfg
        self.args = args

        self.clip_low = args.clip_low
        self.clip_high = args.clip_high
        self.entropy_coeff = args.entropy_coeff
        self.skip_zero_advantage = args.skip_zero_advantage
        self.G = cfg.dapo.num_generations
        self.kl_coeff = cfg.dapo.kl_coeff
        self.advantage_clip = cfg.dapo.advantage_clip
        self.length_penalty_coeff = cfg.dapo.length_penalty_coeff
        self.max_new_tokens = cfg.data.max_new_tokens
        self.temperature = cfg.dapo.temperature
        self.top_p = cfg.dapo.top_p

    def _generate_rollouts(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate G completions per prompt. Returns (sequences, new_mask)."""
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=input_ids.repeat_interleave(self.G, dim=0),
                attention_mask=attention_mask.repeat_interleave(self.G, dim=0),
                max_new_tokens=self.max_new_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        return outputs

    @torch.no_grad()
    def _ref_log_probs(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Per-token log probs under reference model."""
        logits = self.ref_model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits[:, :-1]
        log_probs = F.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(
            dim=2,
            index=input_ids[:, 1:].unsqueeze(-1),
        ).squeeze(-1)
        return token_log_probs

    def train_step(
        self,
        batch: dict,
        optimizer: torch.optim.Optimizer,
        step: int,
    ) -> dict:
        prompts = batch["prompt"]
        ground_truths = batch["ground_truth"]

        enc = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.cfg.data.max_prompt_length,
        ).to(next(self.model.parameters()).device)

        prompt_ids = enc["input_ids"]
        prompt_mask = enc["attention_mask"]
        prompt_len = prompt_ids.shape[1]

        # ── Rollout ───────────────────────────────────────────────────────────
        sequences = self._generate_rollouts(prompt_ids, prompt_mask)
        completion_ids = sequences[:, prompt_len:]   # [B*G, T_gen]
        completions = self.tokenizer.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        # ── Rewards ───────────────────────────────────────────────────────────
        repeated_truths = [gt for gt in ground_truths for _ in range(self.G)]
        repeated_prompts = [p for p in prompts for _ in range(self.G)]
        rewards_list = verifiable_reward(
            prompts=repeated_prompts,
            completions=completions,
            ground_truths=repeated_truths,
            length_penalty_coeff=self.length_penalty_coeff,
        )
        rewards = torch.tensor(
            rewards_list, dtype=torch.float32, device=prompt_ids.device
        )

        # ── Per-group advantage with dynamic sampling ─────────────────────────
        B = len(prompts)
        advantages = torch.zeros_like(rewards)
        groups_skipped = 0
        active_mask = torch.zeros(B * self.G, dtype=torch.bool, device=rewards.device)

        for i in range(B):
            g_rewards = rewards[i * self.G : (i + 1) * self.G]
            adv, is_zero = compute_group_advantages(g_rewards, clip_range=self.advantage_clip)
            if is_zero and self.skip_zero_advantage:
                groups_skipped += 1
                continue
            advantages[i * self.G : (i + 1) * self.G] = adv
            active_mask[i * self.G : (i + 1) * self.G] = True

        if active_mask.sum() == 0:
            logger.warning(f"Step {step}: all groups had zero advantage, skipping step.")
            return {
                "policy_loss": 0.0,
                "entropy": 0.0,
                "kl_div": 0.0,
                "mean_reward": rewards.mean().item(),
                "reward_std": rewards.std().item(),
                "groups_skipped": groups_skipped,
            }

        # ── Build attention mask for completions ──────────────────────────────
        comp_mask = (completion_ids != self.tokenizer.pad_token_id).long()

        # ── Old log probs (under current model before update) ─────────────────
        with torch.no_grad():
            old_logits = self.model(
                input_ids=completion_ids,
                attention_mask=comp_mask,
            ).logits[:, :-1]
            old_log_probs_all = F.log_softmax(old_logits, dim=-1).gather(
                dim=2,
                index=completion_ids[:, 1:].unsqueeze(-1),
            ).squeeze(-1)

        # ── Forward pass for current log probs + logits for entropy ───────────
        logits = self.model(
            input_ids=completion_ids,
            attention_mask=comp_mask,
        ).logits

        log_probs_all = F.log_softmax(logits[:, :-1], dim=-1).gather(
            dim=2,
            index=completion_ids[:, 1:].unsqueeze(-1),
        ).squeeze(-1)

        # Restrict to active (non-skipped) sequences
        active_idx = active_mask.nonzero(as_tuple=True)[0]
        log_probs = log_probs_all[active_idx]
        old_log_probs = old_log_probs_all[active_idx]
        active_adv = advantages[active_idx]
        active_comp_mask = comp_mask[active_idx, 1:]

        # ── DAPO policy loss (token-level, asymmetric clip) ───────────────────
        policy_loss = dapo_policy_loss(
            log_probs=log_probs,
            old_log_probs=old_log_probs,
            advantages=active_adv,
            attention_mask=active_comp_mask,
            clip_low=self.clip_low,
            clip_high=self.clip_high,
        )

        # ── Entropy bonus ─────────────────────────────────────────────────────
        entropy = compute_entropy(logits[active_idx], comp_mask[active_idx, 1:])

        # ── KL divergence against reference ───────────────────────────────────
        ref_log_probs = self._ref_log_probs(completion_ids, comp_mask)
        kl_div = (
            (log_probs_all - ref_log_probs)
            * comp_mask[:, 1:].float()
        ).sum() / comp_mask[:, 1:].float().sum().clamp(min=1)

        total_loss = policy_loss - self.entropy_coeff * entropy + self.kl_coeff * kl_div.clamp(min=0)

        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.cfg.training.max_grad_norm
        )
        optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "entropy": entropy.item(),
            "kl_div": kl_div.item(),
            "mean_reward": rewards.mean().item(),
            "reward_std": rewards.std().item(),
            "groups_skipped": groups_skipped,
        }


# ── Model loading ─────────────────────────────────────────────────────────────
def load_model_and_tokenizer(cfg, base_checkpoint: Optional[str] = None):
    model_name = base_checkpoint or cfg.model.name

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
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

    # Reference model: frozen copy of the initial policy
    ref_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=getattr(torch, cfg.model.torch_dtype),
        attn_implementation=cfg.model.attn_implementation,
        device_map="auto",
    )
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad_(False)

    return model, ref_model, tokenizer


# ── Dataset ───────────────────────────────────────────────────────────────────
def load_data(cfg):
    ds = load_dataset(cfg.data.dataset_id)
    train_ds = ds[cfg.data.train_split]
    eval_ds = ds[cfg.data.eval_split]

    def format_example(ex):
        return {
            "prompt": (
                "Solve the following math problem step by step. "
                "Box your final answer with \\boxed{}.\n\n"
                f"Problem: {ex[cfg.data.problem_field]}\n\nSolution:"
            ),
            "ground_truth": ex[cfg.data.solution_field],
        }

    train_ds = train_ds.map(format_example)
    eval_ds = eval_ds.map(format_example)
    return train_ds, eval_ds


# ── Main ──────────────────────────────────────────────────────────────────────
def main(args: argparse.Namespace) -> None:
    cfg = OmegaConf.load(args.config)

    # CLI overrides take precedence over config file
    cfg.dapo.clip_low = args.clip_low
    cfg.dapo.clip_high = args.clip_high
    cfg.dapo.entropy_coeff = args.entropy_coeff

    wandb.init(
        project=cfg.logging.project,
        name=cfg.logging.run_name,
        config={
            **OmegaConf.to_container(cfg, resolve=True),
            "clip_low": args.clip_low,
            "clip_high": args.clip_high,
            "entropy_coeff": args.entropy_coeff,
            "skip_zero_advantage": args.skip_zero_advantage,
        },
    )

    model, ref_model, tokenizer = load_model_and_tokenizer(cfg, args.base_checkpoint)
    train_ds, eval_ds = load_data(cfg)

    trainer = DAPOTrainer(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        cfg=cfg,
        args=args,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=getattr(cfg.training, "weight_decay", 0.0),
    )

    out_dir = Path("checkpoints/dapo")
    out_dir.mkdir(parents=True, exist_ok=True)

    batch_size = cfg.training.per_device_train_batch_size
    num_epochs = cfg.training.num_train_epochs
    log_every = cfg.logging.logging_steps
    save_every = cfg.training.save_steps

    global_step = 0
    logger.info("Starting DAPO training...")

    for epoch in range(num_epochs):
        indices = torch.randperm(len(train_ds)).tolist()

        for batch_start in range(0, len(indices), batch_size):
            batch_idx = indices[batch_start : batch_start + batch_size]
            batch = {
                "prompt": [train_ds[i]["prompt"] for i in batch_idx],
                "ground_truth": [train_ds[i]["ground_truth"] for i in batch_idx],
            }

            metrics = trainer.train_step(batch, optimizer, global_step)

            if global_step % log_every == 0:
                log_payload = {
                    "train/mean_reward": metrics["mean_reward"],
                    "train/reward_std": metrics["reward_std"],
                    "train/kl_div": metrics["kl_div"],
                    "train/policy_loss": metrics["policy_loss"],
                    "train/entropy": metrics["entropy"],
                    "train/groups_skipped": metrics["groups_skipped"],
                    "train/epoch": epoch,
                    "train/step": global_step,
                }
                wandb.log(log_payload, step=global_step)
                logger.info(
                    f"step={global_step} "
                    f"reward={metrics['mean_reward']:.3f}±{metrics['reward_std']:.3f} "
                    f"kl={metrics['kl_div']:.4f} "
                    f"loss={metrics['policy_loss']:.4f} "
                    f"entropy={metrics['entropy']:.4f} "
                    f"skipped={metrics['groups_skipped']}"
                )

            if global_step > 0 and global_step % save_every == 0:
                ckpt_path = out_dir / f"step_{global_step}"
                model.save_pretrained(str(ckpt_path))
                tokenizer.save_pretrained(str(ckpt_path))
                logger.info(f"Checkpoint saved to {ckpt_path}")

            global_step += 1

    final_path = out_dir / "final"
    model.save_pretrained(str(final_path))
    tokenizer.save_pretrained(str(final_path))
    logger.info(f"DAPO model saved to {final_path}")

    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/dapo_config.yaml")
    parser.add_argument(
        "--base_checkpoint",
        default=None,
        help="Path or HF hub id to load from (DPO or GRPO checkpoint). "
             "Overrides model.name in config.",
    )
    parser.add_argument("--clip_low", type=float, default=0.2)
    parser.add_argument("--clip_high", type=float, default=0.28)
    parser.add_argument("--entropy_coeff", type=float, default=0.001)
    parser.add_argument(
        "--skip_zero_advantage",
        action="store_true",
        default=True,
        help="Skip rollout groups where all rewards are identical (zero advantage).",
    )
    parser.add_argument("--deepspeed", default="configs/ds_zero2.json")
    main(parser.parse_args())
