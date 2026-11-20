.PHONY: doctor setup-local setup-cloud collect-pilot process-pilot annotate-pilot resume report status review-status audit
doctor:
	@scripts/check_local.sh
setup-local:
	@scripts/setup_local.sh
setup-cloud:
	@echo '本项目使用本地推理。'
collect-pilot:
	@scripts/run_pilot.sh collect
process-pilot:
	@scripts/run_pilot.sh process
annotate-pilot:
	@scripts/run_pilot.sh annotate
resume:
	@scripts/run_pilot.sh resume
report status:
	@.venv/bin/python scripts/report.py
review-status:
	@.venv/bin/python scripts/review.py --verify
audit:
	@.venv/bin/python scripts/report.py --audit

.PHONY: test import-review split-reviewed check-layout-dataset train-layout check-xiaomi-finetune train-repair predict-repair evaluate compare-experiments
test:
	@.venv/bin/python -m unittest discover -s tests -v
import-review:
	@.venv/bin/python -m scripts.ground_truth import "$(DECISIONS)"
split-reviewed:
	@.venv/bin/python -m scripts.ground_truth split --version "$(VERSION)"
check-layout-dataset train-layout:
	@$(or $(PADDLE_PYTHON),.venv/bin/python) -m scripts.future_training layout --paddlex "$(PADDLEX)" --dataset "$(DATASET)" --output "$(OUTPUT)" $(if $(filter train-layout,$@),--train --weights "$(WEIGHTS)",)
check-xiaomi-finetune:
	@echo 'Xiaomi LoRA 已归入历史档案。当前研究主线使用原始 BF16 Xiaomi 和 VSSR。' >&2
	@false
train-repair:
	@$(or $(TORCH_PYTHON),../TDA/.venv/bin/python) -m src.training.repair train --dataset "$(DATASET)" --output "$(OUTPUT)" --device "$(or $(DEVICE),cpu)"
predict-repair:
	@$(or $(TORCH_PYTHON),../TDA/.venv/bin/python) -m src.training.repair predict --input "$(INPUT)" --output "$(OUTPUT)" --checkpoint "$(CHECKPOINT)" --device "$(or $(DEVICE),cpu)"
evaluate:
	@.venv/bin/python -m scripts.evaluate evaluate --dataset "$(DATASET)" --config "$(CONFIG)" --predictions "$(PREDICTIONS)" --output "$(OUTPUT)" --freeze "$(FREEZE)" --split "$(or $(SPLIT),validation)"
compare-experiments:
	@.venv/bin/python -m scripts.compare_experiments $(METRICS) --output "$(OUTPUT)"


.PHONY: baseline freeze-evaluation train-xiaomi
baseline: PREDICTIONS ?= data/tables/predictions
baseline: evaluate
freeze-evaluation:
	@.venv/bin/python -m scripts.evaluate freeze --dataset "$(DATASET)" --config "$(CONFIG)" --predictions "$(PREDICTIONS)" --output "$(OUTPUT)" --freeze "$(FREEZE)" --split "$(or $(SPLIT),validation)"
train-xiaomi: check-xiaomi-finetune

.PHONY: setup-layout-gpu
setup-layout-gpu:
	@scripts/setup_layout_gpu.sh

.PHONY: test-vssr check-repvit smoke-vssr
VSSR_PYTHON ?= ../TDA/.venv/bin/python
test-vssr:
	@$(VSSR_PYTHON) -m unittest discover -s tests/vssr -v
check-repvit:
	@$(VSSR_PYTHON) -m scripts.check_repvit $(VSSR_ARGS)
smoke-vssr:
	@$(VSSR_PYTHON) -m scripts.smoke_vssr_all $(VSSR_ARGS)
