install: SHELL:=/bin/bash
install:
	( \
		if ! command -v uv >/dev/null 2>&1; then \
			echo '==> UV not found. Installing...'; \
			curl -LsSf https://astral.sh/uv/install.sh | sh; \
		fi; \
		deactivate || uv venv --python 3.9 --python-preference only-managed; \
		source .venv/bin/activate; \
		uv pip install -r requirements.txt; \
	)

update-deps:
	@uv pip compile pyproject.toml --output-file requirements.txt
	@uv pip compile pyproject.toml --extra bot --output-file bot/requirements.txt
	@uv pip compile pyproject.toml --extra asr-service --output-file inference-server/requirements.txt
	@uv pip compile pyproject.toml --extra train --output-file raw/requirements.txt

pre-commit:
	@pre-commit run --all-files
