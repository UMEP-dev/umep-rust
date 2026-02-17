"""Lightweight checks for user-facing docs consistency."""

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _read(rel_path: str) -> str:
    return (REPO_ROOT / rel_path).read_text(encoding="utf-8")


def test_basic_usage_does_not_claim_isotropic_default():
    """Basic usage should not claim isotropic is the default behavior."""
    text = _read("docs/guide/basic-usage.md")
    assert "By default, SOLWEIG treats diffuse sky radiation as uniform (isotropic)." not in text
    assert "use_anisotropic_sky=True" in text
    assert "MissingPrecomputedData" in text


def test_timeseries_docs_describe_timestep_outputs():
    """Timeseries guide should document the timestep_outputs parameter."""
    text = _read("docs/guide/timeseries.md")
    assert "timestep_outputs" in text


def test_timeseries_docs_describe_report_and_plot():
    """Timeseries guide should document report() and plot()."""
    text = _read("docs/guide/timeseries.md")
    assert ".report()" in text
    assert ".plot(" in text
    assert "summary.timeseries" in text


def test_readme_uses_report_method():
    """README should use summary.report() not print(summary)."""
    text = _read("README.md")
    assert "summary.report()" in text


def test_index_and_readme_state_svf_and_aniso_preconditions():
    """Landing docs should state SVF and explicit anisotropic prerequisites."""
    index_text = _read("docs/index.md")
    readme_text = _read("README.md")

    assert "SVF Rule" in index_text
    assert "Anisotropic Rule" in index_text
    assert "requires SVF to already be prepared" in readme_text
    assert "use_anisotropic_sky=True" in readme_text
