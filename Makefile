.PHONY: setup lint format test run

data:
	uv run python scripts/download_data.py

setup:
	uv sync

lint:
	uv run ruff check . --fix

format:
	uv run ruff format .

test:
	uv run pytest

run:
	uv run honest-model
