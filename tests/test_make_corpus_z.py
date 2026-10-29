"""Arithmetic tests for the Z corpus builder. No bundle, no torch."""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load():
    path = ROOT / "experiments" / "z" / "make_corpus_Z.py"
    spec = importlib.util.spec_from_file_location("make_corpus_Z", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def z():
    return _load()


def test_y_configuration_derives_the_published_shape(z):
    """Y's published parameters are one valid point, not a hard-coded assert."""
    shape = z.derive_shape(
        targets=[32, 64, 109],
        jitter=[-8, -4, 0, 4, 8],
        rows=8,
        steps=200,
        requested_passes=5,
        gaps=[44, 48, 52, 76, 80, 84, 88, 92, 96, 121, 128],
    )
    assert shape["n_geometries"] == 15
    assert shape["presentations_per_geometry"] == 40
    assert shape["passes_per_row"] == 5
    assert shape["len_base"] == 40


def test_wide_probe_shape(z):
    """Separate probe: 9 geometries, 40 rows, 600 steps, 5 passes. Not a 2x2 cell."""
    shape = z.derive_shape(
        targets=[32, 64, 109],
        jitter=[-8, 0, 8],
        rows=40,
        steps=600,
        requested_passes=5,
    )
    assert shape["n_geometries"] == 9
    assert shape["presentations_per_geometry"] == 200
    assert shape["passes_per_row"] == 5
    assert shape["len_base"] == 120
    assert 600 % shape["len_base"] == 0


def test_narrow_probe_shape(z):
    """Separate probe: 3 geometries, 8 rows, 200 steps, 25 passes. Not matched to the wide probe."""
    shape = z.derive_shape(
        targets=[32, 64, 109],
        jitter=[0],
        rows=8,
        steps=200,
        requested_passes=25,
    )
    assert shape["n_geometries"] == 3
    assert shape["presentations_per_geometry"] == 200
    assert shape["passes_per_row"] == 25
    assert shape["len_base"] == 8


def test_impossible_combination_names_the_steps_required(z):
    # 5 jitter values, 40 rows, 200 steps -> 40 presentations/geometry = 1 pass/row, not 5.
    with pytest.raises(SystemExit) as exc:
        z.derive_shape(
            targets=[32, 64, 109],
            jitter=[-8, -4, 0, 4, 8],
            rows=40,
            steps=200,
            requested_passes=5,
        )
    message = str(exc.value)
    assert "1000" in message
    assert "40" in message
    assert "1" in message


def test_require_bands_skip_is_loud_and_a_mid_jitter_is_not_a_coverage_error(z):
    # [-4, 0, 4] has more than one jitter value and still does not cover 40.
    # Gating coverage on len(jitter) > 1 would reject this build for the wrong reason.
    shape = z.derive_shape(
        targets=[32, 64, 109],
        jitter=[-4, 0, 4],
        rows=8,
        steps=24,
        requested_passes=1,
    )
    assert 40 not in shape["geometries"]
    assert 72 not in shape["geometries"]
    skipped = z.coverage_status(shape["geometries"], [])
    assert skipped == "BAND_COVERAGE skipped (REQUIRE_BANDS empty)"
    with pytest.raises(SystemExit) as exc:
        z.coverage_status(shape["geometries"], [40])
    assert "40" in str(exc.value)
    assert "not covered" in str(exc.value)


def test_require_bands_passes_when_the_distance_is_actually_present(z):
    shape = z.derive_shape(
        targets=[32, 64, 109],
        jitter=[-8, -4, 0, 4, 8],
        rows=8,
        steps=200,
        requested_passes=5,
    )
    assert z.coverage_status(shape["geometries"], [40, 72]) == "BAND_COVERAGE ok: 40,72"


def test_missing_probe_file_is_fail_closed(z):
    missing = "/tmp/does-not-exist-probe_Q" + ".pt"
    with pytest.raises(SystemExit) as exc:
        z.probe_payloads(missing)
    assert "probe_Q.pt" in str(exc.value)
    assert "refusing" in str(exc.value)


def test_default_require_bands_cli_is_empty(z, tmp_path, monkeypatch, capsys):
    monkeypatch.delenv("Z_REQUIRE_BANDS", raising=False)
    # Force the build to stop at the missing source, after the parameter prints.
    monkeypatch.setenv("T2_SRC", str(tmp_path / "no-such-src"))
    with pytest.raises(SystemExit) as exc:
        z.main([])
    out = capsys.readouterr().out
    assert "DERIVED geometries=15 presentations_per_geometry=40 passes_per_row=5" in out
    assert "BAND_COVERAGE skipped (REQUIRE_BANDS empty)" in out
    assert "run_vyra1_copy.py" in str(exc.value)
