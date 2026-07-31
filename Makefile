.PHONY: test accuracy blindspots clean

test:
	uv run pytest -q

accuracy:
	uv run python bench/accuracy.py

blindspots:
	uv run python bench/blindspots.py

clean:
	rm -rf .pytest_cache **/__pycache__
