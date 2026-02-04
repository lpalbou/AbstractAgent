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

This repo already contains `dist/` artifacts, but to rebuild locally:

```bash
python -m pip install build
python -m build
```

