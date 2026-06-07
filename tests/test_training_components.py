"""tests/test_training_components.py"""
import torch
import pytest
from training.train_grpo import compute_group_advantages


class TestGroupAdvantages:
    def test_normalisation(self):
        rewards = torch.tensor([1.0, 0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0])
        adv = compute_group_advantages(rewards)
        assert abs(adv.mean().item()) < 1e-5, "Advantages should be zero-mean"

    def test_clipping(self):
        # Extreme outlier should be clipped
        rewards = torch.tensor([100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        adv = compute_group_advantages(rewards, clip_range=5.0)
        assert adv.max().item() <= 5.0, "Advantages should be clipped to clip_range"
        assert adv.min().item() >= -5.0

    def test_normalize_before_clip(self):
        """Verify we normalise FIRST then clip — not the reverse."""
        rewards = torch.tensor([1.0, -1.0, 1.0, -1.0, 0.5, -0.5, 0.5, -0.5])
        adv = compute_group_advantages(rewards, clip_range=5.0)
        # With clip_range=5 and typical rewards, no values should be clipped
        std_adv = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        assert torch.allclose(adv, std_adv.clamp(-5.0, 5.0))

    def test_all_same_rewards(self):
        """All same rewards → zero advantages (avoid division by zero)."""
        rewards = torch.ones(8)
        adv = compute_group_advantages(rewards)
        assert torch.all(adv == 0.0) or torch.all(adv.abs() < 1e-4)

    def test_shape_preserved(self):
        rewards = torch.randn(8)
        adv = compute_group_advantages(rewards)
        assert adv.shape == rewards.shape


class TestDPOLoss:
    """Sanity checks for DPO loss behaviour."""

    def _mock_dpo_loss(self, chosen_logp: float, rejected_logp: float, beta: float = 0.1):
        """Simplified DPO loss for unit testing."""
        import torch.nn.functional as F
        chosen = torch.tensor(chosen_logp)
        rejected = torch.tensor(rejected_logp)
        loss = -F.logsigmoid(beta * (chosen - rejected))
        return loss.item()

    def test_correct_preference_reduces_loss(self):
        """Higher chosen log-prob than rejected should give lower loss."""
        loss_good = self._mock_dpo_loss(chosen_logp=-1.0, rejected_logp=-2.0)
        loss_bad = self._mock_dpo_loss(chosen_logp=-2.0, rejected_logp=-1.0)
        assert loss_good < loss_bad

    def test_loss_is_positive(self):
        loss = self._mock_dpo_loss(-1.0, -2.0)
        assert loss > 0

    def test_larger_beta_amplifies_margin(self):
        """Higher β → more sensitive to reward margin."""
        loss_small_beta = self._mock_dpo_loss(-1.0, -2.0, beta=0.05)
        loss_large_beta = self._mock_dpo_loss(-1.0, -2.0, beta=0.2)
        # With a good margin (chosen > rejected), larger beta → smaller loss
        assert loss_large_beta < loss_small_beta
