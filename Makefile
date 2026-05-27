PYTHON ?= python3
UV ?= uv

.PHONY: install test smoke lint format clean

install:
	$(UV) venv --python 3.11
	$(UV) pip install -e ".[dev]"

install-heavy:
	$(UV) pip install -e ".[dev,heavy]"

test:
	$(UV) run pytest tests/ -x

smoke:
	bash scripts/smoke_test.sh

lint:
	$(UV) run ruff check src tests
	$(UV) run black --check src tests

format:
	$(UV) run ruff check --fix src tests
	$(UV) run black src tests

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache
	rm -rf build dist *.egg-info src/*.egg-info
	rm -rf outputs lightning_logs
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
