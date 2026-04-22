We use [uv] to develop harmonypy.

Copy the harmonypy code to your computer:

```
git clone https://github.com/slowkow/harmonypy
```

Then change to the newly created directory:

```
cd harmonypy
```

Install uv:

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create a virtual environment and install harmonypy in editable mode with test
dependencies:

```
uv venv
uv pip install -e ".[test]"
```

Run the tests:

```
uv run pytest tests/test_harmony.py -v
```

Build the wheel:

```
uv pip install build
python -m build
```

Wheels are published to PyPI automatically via GitHub Actions when a new
release is created on GitHub. See `.github/workflows/release.yml`.

[uv]: https://docs.astral.sh/uv/
