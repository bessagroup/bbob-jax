# Makefile for Python project using uv, pytest, and mkdocs

# Default goal
.DEFAULT_GOAL := test

# Variables
PYTEST_CMD = uv run --no-sync pytest
MKDOCS_CMD = uv run --no-sync mkdocs serve

# Run tests
test:
	@echo "Running tests with pytest..."
	@$(PYTEST_CMD)

# Run documentation server
docs:
	@echo "Starting MkDocs development server..."
	@$(MKDOCS_CMD)

# Run linter
lint:
	@echo "Running Ruff linter..."
	@uv run --no-sync ruff check .

# Format code
format:
	@echo "Formatting code with Ruff..."
	@uv run --no-sync ruff format .

# Clean build artifacts
clean:
	@echo "Cleaning up..."
	@rm -rf __pycache__ .pytest_cache .ruff_cache build dist site *.egg-info

.PHONY: test docs lint format clean
