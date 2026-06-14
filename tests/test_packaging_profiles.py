from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore


ROOT = Path(__file__).resolve().parents[1]


def _project() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def test_base_dependency_floors_match_gateway_alignment() -> None:
    deps = set(_project()["dependencies"])

    assert "abstractcore[tools]>=2.13.38" in deps
    assert "abstractruntime>=0.4.29" in deps


def test_hardware_profile_extras_are_core_runtime_cascades() -> None:
    extras = _project()["optional-dependencies"]

    assert set(extras["apple"]) == {
        "abstractcore[all-apple]>=2.13.38",
        "abstractruntime[apple]>=0.4.29",
    }
    assert set(extras["gpu"]) == {
        "abstractcore[all-gpu]>=2.13.38",
        "abstractruntime[gpu]>=0.4.29",
    }
    assert "all-apple" not in extras
    assert "all-gpu" not in extras
