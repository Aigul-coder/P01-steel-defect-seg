.PHONY: install data-synth train-baseline train-improved eval onnx test lint api compose

install:
	pip install -U pip
	pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
	pip install -e ".[dev]"

data-synth:
	python scripts/download_data.py --source synthetic --subset 64

train-baseline:
	python scripts/train.py --config configs/baseline.yaml

train-improved:
	python scripts/train.py --config configs/improved.yaml

eval:
	python scripts/eval.py --config configs/improved.yaml

onnx:
	python scripts/export_onnx.py --config configs/improved.yaml

test:
	pytest -q

lint:
	ruff check src tests scripts
	ruff format --check src tests scripts

api:
	uvicorn steel_defect.api:app --host 0.0.0.0 --port 8000

compose:
	docker compose -f docker/docker-compose.yml up --build
