# verifiable-alignment

**Full post-training pipeline: DPO → GRPO → RLVR on mathematical reasoning**

[![CI](https://github.com/saitejasrivilli/verifiable-alignment/actions/workflows/ci.yml/badge.svg)](https://github.com/saitejasrivilli/verifiable-alignment/actions)
[![W&B](https://img.shields.io/badge/Weights%20%26%20Biases-Dashboard-orange)](https://wandb.ai/utarlington/verifiable-alignment)
[![HuggingFace](https://img.shields.io/badge/🤗-Model%20Hub-yellow)](https://huggingface.co/SaiTejaSrivilli/verifiable-alignment-grpo)
[![Demo](https://img.shields.io/badge/🚀-Live%20Demo-blue)](https://huggingface.co/spaces/SaiTejaSrivilli/verifiable-alignment)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

---

## Overview

This project implements the complete modern post-training stack on a 7B language model,
covering every stage that frontier labs (Anthropic, OpenAI, DeepSeek) use in production:

| Stage | Method | Key result |
|---|---|---|
| Synthetic data | Self-play, temperature sweep, difficulty stratification | 10K preference pairs |
| Preference optimisation | DPO (β=0.1, LoRA rank-64) | +7.3% MATH accuracy |
| RL with verifiable rewards | GRPO (G=8, symbolic verifier) | +6.6% over DPO |
| RLAIF | AI judge (Groq) → DPO on AI-generated preference pairs | scalable alternative to RLHF |
| Constitutional AI | Self-critique against 12 principles → revision pairs → DPO | no human/verifier needed |
| Test-time compute | PRM best-of-8 reranking | +4.2% over GRPO |
| Reward hacking analysis | Pearson(reward, accuracy), length exploitation, format gaming | diagnostic |

**Final: Mistral-7B base 12.4% → 30.5% on MATH** (×2.46 improvement)

### Feedback signal comparison

| Method | Label source | Scale | Cost | Coverage |
|---|---|---|---|---|
| RLHF | Humans | ~1K pairs | High | Subjective quality |
| RLAIF | AI judge (Groq/Claude) | Unlimited | Low (API) | Any task |
| RLVR | Symbolic verifier | Unlimited | Zero | Verifiable tasks only |
| Constitutional AI | Model self-critique | Unlimited | Zero | Any task |

---

## Results

| Model | MATH (overall) | MATH L4–5 | GSM8K |
|---|---|---|---|
| Mistral-7B base | 12.4% | 4.1% | 41.2% |
| + DPO (synthetic data) | 19.7% | 8.3% | 58.9% |
| + GRPO (verifiable rewards) | 26.3% | 13.7% | 71.4% |
| + PRM reranking (N=8) | **30.5%** | **16.2%** | **74.1%** |

MATH evaluation: 4-shot via [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness) (`minerva_math` task).
GSM8K evaluation: 8-shot chain-of-thought, strict-match scoring.

---

## DAPO Variant

DAPO addresses instability on hard problems (MATH Level 4–5) via four changes over GRPO:

| Change | GRPO | DAPO | Why |
|--------|------|------|-----|
| Clip range | symmetric ε=0.2 | low=0.2, high=0.28 | Larger positive updates improve hard problems |
| Gradient | sequence-level | token-level | Lower variance for long CoT chains |
| Zero-advantage groups | included (zero gradient) | skipped | Removes noise from trivial batches |
| Entropy | none | coeff=0.001 | Prevents mode collapse on hard problems |

Expected improvement on MATH Level 4–5: +3–5% over standard GRPO (run pending on 4×A30).

---

## Architecture

```
Mistral-7B-v0.3 (base)
        │
        ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 1 — Synthetic Preference Dataset                  │
│  • Gemma-2-2B teacher, T ∈ {0.3, 0.7, 1.0}             │
│  • Symbolic verifier (sympy exact-match)                 │
│  • Difficulty stratification: MATH levels 1–5            │
│  • 10K (chosen, rejected) pairs → HuggingFace Hub        │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 2 — DPO Fine-tuning                               │
│  • TRL DPOTrainer + LoRA (r=64, α=128)                  │
│  • β=0.1 KL penalty (swept: 0.05, 0.1, 0.2)            │
│  • Tracks: reward margin, KL divergence, length dist.    │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 3 — GRPO with Verifiable Rewards                  │
│  • G=8 rollouts per problem                              │
│  • Group-normalise THEN clip advantages (correct order)  │
│  • Entropy bonus (0.001) + length regularisation         │
│  • DeepSpeed ZeRO-2, 2×A30                               │
│  • vLLM rollout generation (3–5× speedup)               │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 3b — DAPO (optional GRPO replacement)             │
│  • Asymmetric clip: low=0.2, high=0.28                   │
│  • Token-level policy gradient (lower variance)          │
│  • Dynamic sampling: skip zero-advantage groups          │
│  • Entropy bonus (coeff=0.001) vs mode collapse          │
└──────────────────────────┬──────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────┐
│  Stage 4 — Process Reward Model                          │
│  • MATH-Shepherd step-level annotations                  │
│  • Frozen base + linear head (head-only training)        │
│  • Best-of-8 reranking: cumulative product of step probs │
│  • Calibration diagram logged to W&B                     │
└─────────────────────────────────────────────────────────┘
```

---

## Repository structure

```
verifiable-alignment/
├── configs/
│   ├── dpo_base.yaml          # DPO hyperparameters
│   ├── dpo_sweep.yaml         # W&B sweep (β × LR grid)
│   ├── grpo_base.yaml         # GRPO hyperparameters
│   ├── dapo_config.yaml       # DAPO hyperparameters (asymmetric clip, token-level)
│   ├── prm_base.yaml          # PRM hyperparameters
│   ├── ds_zero2.json          # DeepSpeed ZeRO-2
│   └── accelerate_single.yaml
├── data/
│   ├── generate_preferences.py   # Self-play synthetic data pipeline
│   └── verifier.py               # Symbolic math verifier (sympy)
├── training/
│   ├── train_dpo.py              # DPO + LoRA + KL/length callbacks
│   ├── train_grpo.py             # GRPO + verifiable reward fn
│   ├── train_dapo.py             # DAPO: clip-higher, token-level, dynamic sampling
│   └── train_prm.py              # PRM head + calibration
├── eval/
│   ├── evaluate_math.py          # lm-eval harness, difficulty breakdown
│   ├── reward_hacking.py         # Pearson(RM, verifier) analysis
│   └── kl_divergence.py          # KL–accuracy tradeoff plot
├── demo/
│   └── app.py                    # Gradio HuggingFace Spaces demo
├── tests/
│   ├── test_verifier.py
│   └── test_training_components.py
├── docker/Dockerfile
├── pyproject.toml
└── Makefile
```

---

## Quickstart

```bash
# 1. Install
git clone https://github.com/saitejasrivilli/verifiable-alignment
cd verifiable-alignment
pip install -e ".[dev]"

# 2. Generate synthetic preference data
make data

# 3. DPO fine-tuning (single GPU)
make dpo

# 4. GRPO training (2× GPU, DeepSpeed)
make grpo

# 5. PRM training
make prm

# 6. Full evaluation
make eval-all

# 7. Run demo locally
make demo
```

---

## W&B training dashboard

All training curves are public:
[**→ Open W&B Dashboard**](https://wandb.ai/utarlington/verifiable-alignment)

Key signals logged:

| Signal | What it indicates |
|---|---|
| `train/rewards/margins` | DPO learning preference signal |
| `eval/kl_divergence` | Policy drift from reference (per epoch, fixed 500-sample set) |
| `train/clip_fraction` | GRPO stability (alert if >0.3) |
| `eval/math_accuracy` | Per difficulty level (1–5) breakdown |
| `eval/length_distribution` | Reward hacking early warning |
| `eval/prm_calibration` | PRM reliability diagram |

---

## Training data

**Dataset:** [`SaiTejaSrivilli/verifiable-alignment-preferences`](https://huggingface.co/datasets/SaiTejaSrivilli/verifiable-alignment-preferences)

**Generation procedure:**
- Source problems: GSM8K train (7473) + MATH train (7500)
- Teacher model: `google/gemma-2-2b-it`
- Candidate solutions: 4 per problem across T ∈ {0.3, 0.7, 1.0}
- Scoring: symbolic verifier (sympy exact-match + LaTeX parsing)
- Pair construction: random (correct, incorrect) pair per problem
- Difficulty stratification: MATH level distribution maintained
- Deduplication: MD5 hash on problem text, exact dedup

**Rejection rate:** ~38% of candidates were discarded (no contrast pair possible).

**Known limitations of the dataset:**
- GSM8K problems are assigned difficulty=2 uniformly (approximate)
- Very short problems (<20 tokens) underrepresented in levels 4–5
- Teacher model (2B) may produce lower-quality rejected solutions on hard problems

---

## Known model limitations

- **Domain:** Trained exclusively on school-level and competition math. Performance on
  non-mathematical reasoning tasks is not evaluated and likely degrades from base.
- **Reward hacking:** GRPO model occasionally produces verbose, structured-looking but
  incorrect solutions that score highly on implicit reward. See reward hacking visualiser
  in demo.
- **Hard problems:** MATH levels 4–5 remain challenging (16.2% accuracy with PRM).
  Problems requiring geometric insight or advanced combinatorics are underrepresented.
- **Language:** English only. Multi-lingual math problems not tested.

---

## Reward hacking analysis

The DPO implicit reward (log π_θ/π_ref) has a Pearson correlation of **r=0.61** with
symbolic verifier correctness, meaning ~39% of variance in RM score is unexplained by
actual correctness. Common failure patterns observed:

1. **Verbosity gaming** — padding correct answers with extra explanation inflates log-likelihood
2. **Format mimicry** — structured "Step 1, Step 2, ..." pattern earns reward regardless of content
3. **Confident wrong answers** — high-certainty phrasing inflates reward on incorrect solutions

GRPO with verifiable rewards reduces this to r=0.84 (Pearson), as the symbolic verifier
provides a direct binary signal that cannot be gamed.

---

## Hardware & reproducibility

| Stage | Hardware | Wall time | Cost |
|---|---|---|---|
| Synthetic data | 1× A100 (Colab Pro) | ~3 hrs | $0 (Colab CU) |
| DPO | 1× A30 24GB | ~4 hrs | $0 (own cluster) |
| GRPO | 2× A30 24GB, ZeRO-2 | ~6 hrs | $0 (own cluster) |
| PRM | 1× A30 24GB | ~3 hrs | $0 (own cluster) |
| Evaluation | 1× A100 (Colab Pro) | ~2 hrs | $0 (Colab CU) |

All random seeds are fixed. Docker image available for full environment reproduction:
```bash
docker pull ghcr.io/saitejasrivilli/verifiable-alignment:latest
```

---

## Citation

```bibtex
@misc{verifiable-alignment-2025,
  title  = {verifiable-alignment: DPO → GRPO → RLVR Post-Training Pipeline},
  author = {Srivillibhutturu Saiteja},
  year   = {2025},
  url    = {https://github.com/saitejasrivilli/verifiable-alignment}
}
```

---

## License

Apache 2.0. See [LICENSE](LICENSE).
# PostTrain-7B
