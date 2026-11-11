.PHONY: test demo eval lint run clean

PYTHON ?= python3
export PYTHONPATH := src

test:
	$(PYTHON) -m pytest -q

demo:
	$(PYTHON) -m convai.cli --demo

eval:
	$(PYTHON) -m convai.cli --eval

run:
	$(PYTHON) -m convai.service

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
