# Spy Test Suite - Development Commands
.PHONY: help install test lint format clean docs

# Default target
help:
	@echo "Spy Test Suite Development Commands"
	@echo "=================================="
	@echo "install    - Install all dependencies"
	@echo "test       - Run all tests"
	@echo "lint       - Run linting and type checking"
	@echo "format     - Format code with black and isort"
	@echo "clean      - Clean build artifacts and cache"
	@echo "docs       - Generate documentation"
	@echo "ci         - Run CI pipeline locally"

# Install dependencies
install:
	@echo "Installing dependencies..."
	pip install -e .
	pip install -e ".[dev,docs,benchmark]"
	pre-commit install

# Run tests
test:
	@echo "Running tests..."
	pytest --cov=spy_test_suite --cov-report=html --cov-report=term-missing -v

# Run specific test types
test-unit:
	pytest tests/unit/ -v

test-integration:
	pytest tests/integration/ -v --tb=short

test-performance:
	pytest tests/performance/ -v -m performance

test-statistical:
	pytest tests/statistical/ -v -m statistical

# Code quality
lint:
	@echo "Running linting..."
	flake8 spy_test_suite tests
	mypy spy_test_suite
	@echo "Linting complete!"

format:
	@echo "Formatting code..."
	black spy_test_suite tests
	isort spy_test_suite tests
	@echo "Code formatted!"

# Clean up
clean:
	@echo "Cleaning up..."
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name .mypy_cache -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -type d -name .coverage -exec rm -rf {} +
	find . -type d -name htmlcov -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name ".coverage" -delete
	find . -type f -name "coverage.xml" -delete
	@echo "Cleanup complete!"

# Documentation
docs:
	@echo "Generating documentation..."
	sphinx-build -b html docs/source docs/build/html
	@echo "Documentation generated in docs/build/html/"

# CI pipeline
ci: lint test
	@echo "CI pipeline passed!"

# Development server
dev:
	@echo "Starting development environment..."
	jupyter notebook --notebook-dir=. --ip=0.0.0.0 --port=8888 --no-browser

# Package management
build:
	@echo "Building packages..."
	python -m build

publish-test:
	@echo "Publishing to TestPyPI..."
	python -m twine upload --repository testpypi dist/*

publish:
	@echo "Publishing to PyPI..."
	python -m twine upload dist/*

# Git operations
commit: format lint test
	@echo "Ready to commit! All checks passed."

# Performance profiling
profile:
	@echo "Running performance profiling..."
	python -m cProfile -s cumulative scripts/profile_main.py

# Benchmarking
benchmark:
	@echo "Running benchmarks..."
	python scripts/benchmark.py

# Docker
docker-build:
	docker build -t spy-test-suite .

docker-run:
	docker run -it --rm spy-test-suite

# Help for specific packages
help-fairness:
	@echo "Fairness Tests Package"
	@echo "======================"
	@echo "cd packages/fairness-tests"
	@echo "pip install -e ."
	@echo "python -m pytest tests/"

help-visualization:
	@echo "Visualization Package"
	@echo "====================="
	@echo "cd packages/visualization"
	@echo "pip install -e ."
	@echo "python scripts/generate_plots.py"

help-benchmark:
	@echo "Benchmark Package"
	@echo "================="
	@echo "cd packages/benchmark"
	@echo "pip install -e ."
	@echo "python scripts/run_benchmarks.py"

help-integration:
	@echo "Integration Package"
	@echo "==================="
	@echo "cd packages/integration"
	@echo "pip install -e ."
	@echo "python -m pytest tests/"