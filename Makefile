install: SHELL:=/bin/bash
install:
	@if [[ -z ${VIRTUAL_ENV} ]]; then\
		@uv --version || curl -LsSf https://astral.sh/uv/install.sh | sh;\
		@uv venv --python 3.11 --python-preference only-managed;\
		@source .venv/bin/activate;\
	fi
	@uv pip install -r requirements.txt

update-deps:
	@uv pip compile pyproject.toml --extra bot --output-file bot/requirements.txt
	@uv pip compile pyproject.toml --extra asr-service --output-file inference-server/requirements.txt
	@uv pip compile pyproject.toml --extra train --output-file raw/requirements.txt

pre-commit:
	@pre-commit run --all-files
