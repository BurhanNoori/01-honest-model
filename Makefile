.PHONY: setup lint format test run

setup:
	uv sync

lint:
	uv run ruff check .

format:
	uv run ruff format .

test:
	uv run pytest

run:
	uv run honest-model