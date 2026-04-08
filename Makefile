# LLM-Brain Development Commands
# Usage: make <command>

.PHONY: help install install-dev test test-cov lint format type-check check-all clean setup

PYTHON := python3
PIP := pip3

help: ## Show this help message
	@echo "LLM-Brain Development Commands"
	@echo "=============================="
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

# =============================================================================
# Installation
# =============================================================================

install: ## Install package in production mode
	$(PIP) install -e .

install-dev: ## Install package with all dev dependencies
	$(PIP) install -e ".[dev]"
	pre-commit install

setup: install-dev ## Full setup for new developers
	@echo "✓ Development environment ready"
	@echo "Run 'make check-all' to verify everything works"

# =============================================================================
# Testing
# =============================================================================

test: ## Run tests
	pytest tests/ -v

test-cov: ## Run tests with coverage report
	pytest tests/ --cov=llm_brain --cov-report=term-missing --cov-report=html

test-cov-fail: ## Run tests with coverage threshold enforcement
	pytest tests/ --cov=llm_brain --cov-fail-under=80

# =============================================================================
# Code Quality
# =============================================================================

lint: ## Run ruff linter with auto-fix
	ruff check src/ tests/ --fix

lint-check: ## Run ruff linter without fixing (CI mode)
	ruff check src/ tests/

format: ## Format code with ruff
	ruff format src/ tests/

format-check: ## Check formatting without modifying (CI mode)
	ruff format src/ tests/ --check

type-check: ## Run mypy type checker
	mypy src/llm_brain

type-check-strict: ## Run mypy with strict mode
	mypy src/llm_brain --strict

check-all: lint-check format-check type-check test-cov ## Run all checks (CI mode)

# =============================================================================
# Pre-commit
# =============================================================================

pre-commit: ## Run all pre-commit hooks on all files
	pre-commit run --all-files

pre-commit-update: ## Update pre-commit hooks to latest versions
	pre-commit autoupdate

# =============================================================================
# Maintenance
# =============================================================================

clean: ## Clean build artifacts and cache files
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name ".coverage.*" -delete 2>/dev/null || true

# =============================================================================
# Build & Publish (when ready)
# =============================================================================

build: clean ## Build package distribution
	$(PYTHON) -m build

publish-test: build ## Publish to TestPyPI
	$(PYTHON) -m twine upload --repository testpypi dist/*

publish: build ## Publish to PyPI
	$(PYTHON) -m twine upload dist/*

# =============================================================================
# Utilities
# =============================================================================

bootstrap: ## Run bootstrap script
	$(PYTHON) scripts/bootstrap.py

bootstrap-check: ## Check brain health
	$(PYTHON) scripts/bootstrap.py --check
