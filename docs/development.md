# Development

Related:
- [`README.md`](../README.md)
- [`docs/architecture.md`](architecture.md)

## Setup

```bash
pip install -e ".[dev]"
```

## Run tests

```bash
pytest
```

## Build distributions (optional)

This repo may contain `dist/` artifacts for convenience, but they can be stale. To rebuild locally (recommended before publishing):

```bash
python -m pip install build
python -m build
```
