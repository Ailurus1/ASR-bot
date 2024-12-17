SHELL:=/bin/bash

install: setup-uv setup-deps

setup-uv:
	@if ! command -v uv >/dev/null 2>&1; then \
		echo '==> UV not found. Installing...' && \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	else \
		echo 'UV is already installed'; \
	fi

setup-deps:
	@if [ -d ".venv" ]; then \
		rm -rf .venv; \
	fi
	uv venv --python 3.11 --python-preference only-managed && \
	source .venv/bin/activate && \
	uv pip install -r requirements.txt

pre-commit:
	@pre-commit run --all-files
