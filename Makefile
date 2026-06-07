.PHONY: install lint test data dpo grpo prm eval demo clean

install:
	pip install -e ".[dev]"
	pre-commit install

lint:
	ruff check . --fix
	mypy training/ eval/ data/

test:
	pytest tests/ -v --cov=. --cov-report=html

# ── Data ──────────────────────────────────────────────────────────────────────
data:
	python data/generate_preferences.py \
		--config configs/data_gen.yaml \
		--push_to_hub

# ── Training stages ───────────────────────────────────────────────────────────
dpo:
	accelerate launch --config_file configs/accelerate_single.yaml \
		training/train_dpo.py \
		--config configs/dpo_base.yaml

dpo-sweep:
	wandb sweep configs/dpo_sweep.yaml

grpo:
	deepspeed --num_gpus=2 training/train_grpo.py \
		--config configs/grpo_base.yaml \
		--deepspeed configs/ds_zero2.json

prm:
	accelerate launch --config_file configs/accelerate_single.yaml \
		training/train_prm.py \
		--config configs/prm_base.yaml

# ── Evaluation ────────────────────────────────────────────────────────────────
eval-all:
	python eval/evaluate_math.py --models base dpo grpo grpo_prm
	python eval/reward_hacking.py
	python eval/kl_divergence.py

eval-quick:
	python eval/evaluate_math.py --models dpo grpo --limit 100

# ── Demo ──────────────────────────────────────────────────────────────────────
demo:
	python demo/app.py

demo-deploy:
	huggingface-cli upload verifiable-alignment/demo demo/ --repo-type=space

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache htmlcov .coverage dist build *.egg-info
