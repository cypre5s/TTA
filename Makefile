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
	@$(or $(XIAOMI_PYTHON),../TDA/.venv/bin/python) -m scripts.future_training xiaomi-probe --rank "$(or $(RANK),8)" --model ../TDA/assets/Xiaomi-OCR-0 --dataset "$(DATASET)" --output "$(OUTPUT)"
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
train-xiaomi:
	@$(or $(XIAOMI_PYTHON),../TDA/.venv/bin/python) -m scripts.future_training xiaomi-train --model ../TDA/assets/Xiaomi-OCR-0 --dataset "$(DATASET)" --output "$(OUTPUT)" --probe "$(PROBE)" --rank "$(or $(RANK),8)" --epochs "$(or $(EPOCHS),1)" --accumulation "$(or $(ACCUMULATION),8)"

.PHONY: setup-layout-gpu
setup-layout-gpu:
	@scripts/setup_layout_gpu.sh
