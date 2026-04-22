PYTHON  ?= python3
SRC     = atlassian_rag_exporter.py auth_providers.py elasticsearch_source.py vector_store.py
TESTS   = tests/

.PHONY: install install-all test test-cov lint fmt typecheck ci clean

install:
	$(PYTHON) -m pip install -e ".[dev]"

install-all:
	$(PYTHON) -m pip install -e ".[dev,kerberos,vector-all]"

test:
	$(PYTHON) -m pytest $(TESTS) -v --tb=short

test-cov:
	$(PYTHON) -m pytest $(TESTS) -v --tb=short \
	    --cov=atlassian_rag_exporter \
	    --cov=auth_providers \
	    --cov=elasticsearch_source \
	    --cov=vector_store \
	    --cov-report=html:docs/htmlcov \
	    --cov-report=term-missing \
	    --cov-fail-under=80

lint:
	$(PYTHON) -m ruff check $(SRC) $(TESTS)

fmt:
	$(PYTHON) -m black $(SRC) $(TESTS)
	$(PYTHON) -m ruff check --fix $(SRC) $(TESTS)

typecheck:
	$(PYTHON) -m mypy $(SRC)

ci: fmt lint typecheck test-cov
	@echo ""
	@echo "\033[32m====================================\033[0m"
	@echo "\033[32m  ALL CI CHECKS PASSED \xE2\x9C\x85\033[0m"
	@echo "\033[32m====================================\033[0m"

clean:
	rm -rf .pytest_cache __pycache__ tests/__pycache__ .mypy_cache htmlcov .coverage docs/htmlcov
