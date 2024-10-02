install: SHELL:=/bin/bash
install:
	@uv --version || curl -LsSf https://astral.sh/uv/install.sh | sh
	@uv venv --python 3.11 --python-preference only-managed
	@source .venv/bin/activate
	@uv pip install -r requirements.txt

update-deps:
	@uv pip compile pyproject.toml --output-file=requirements.txt

pre-commit:
	@pre-commit run --all-files
