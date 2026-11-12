"""Compressed asset embedding and the Kaggle 1 MB notebook-source size gate (tools/experiment_packaging/notebook_pack.py)."""
import base64
import hashlib
import json
import sys
import zlib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools" / "experiment_packaging"))

import notebook_pack as NP  # noqa: E402
import y72_rehearsal2_pool as pool  # noqa: E402

POOL_FILE = ROOT / "prereg" / "y72_rehearsal2" / "y72_fresh_pool_F200_set.json"


def test_limit_is_kaggle_one_megabyte():
    assert NP.KAGGLE_SOURCE_LIMIT == 1048576 == 1 << 20


def test_size_gate_rejects_at_and_over_the_limit_and_accepts_under():
    assert NP.check_size(0) == 0 and NP.check_size(1048575) == 1048575
    for n in (1048576, 1048577, 1708020):                    # 1708020 = the rejected c3ab200b notebook
        with pytest.raises(NP.PackError, match="NOTEBOOK_TOO_LARGE"):
            NP.check_size(n)
    with pytest.raises(NP.PackError, match="NOTEBOOK_SIZE_INVALID"):
        NP.check_size(-1)


def test_size_gate_on_files(tmp_path):
    small, big, exact = tmp_path / "s.ipynb", tmp_path / "b.ipynb", tmp_path / "e.ipynb"
    small.write_bytes(b"x" * 1000)
    big.write_bytes(b"x" * (NP.KAGGLE_SOURCE_LIMIT + 1))
    exact.write_bytes(b"x" * NP.KAGGLE_SOURCE_LIMIT)
    assert NP.check_notebook_file(small) == 1000
    for f in (big, exact):
        with pytest.raises(NP.PackError, match="NOTEBOOK_TOO_LARGE"):
            NP.check_notebook_file(f)
    with pytest.raises(NP.PackError, match="NOTEBOOK_MISSING"):
        NP.check_notebook_file(tmp_path / "absent.ipynb")


def test_pack_is_deterministic_and_lossless():
    assets = {"b.py": "print('b')\n" * 50, "a.json": json.dumps({"k": [1, 2, 3]}), "u.txt": "non-ascii \u00e9\u4e2d\n"}
    blob = NP.pack_assets(assets)
    assert blob == NP.pack_assets(dict(reversed(list(assets.items()))))     # insertion order does not matter
    assert NP.unpack_assets(blob) == assets
    with pytest.raises(NP.PackError):
        NP.pack_assets({"a": b"bytes"})


def test_bootstrap_executes_and_fails_closed_on_a_changed_blob():
    assets = {"x.py": "X=1\n" * 100}
    code = NP.assets_bootstrap(assets)
    ns = {"hashlib": hashlib, "json": json, "base64": base64}
    exec(code, ns)
    assert ns["assets"] == assets
    lines = code.splitlines(keepends=True)
    i = next(k for k, l in enumerate(lines) if l.startswith("assets_z="))
    blob = eval(lines[i][len("assets_z="):])
    lines[i] = "assets_z=" + repr(blob[:-6] + ("A" if blob[-6] != "A" else "B") + blob[-5:]) + "\n"
    with pytest.raises(ValueError, match="BOOTSTRAP_PACK_CORRUPT"):
        exec("".join(lines), {"hashlib": hashlib, "json": json, "base64": base64})
    assert NP.embedded_assets(code) == assets


def test_decompressed_pool_is_the_registered_pool(tmp_path):
    text = POOL_FILE.read_text()
    out = NP.unpack_assets(NP.pack_assets({pool.POOL_ASSET: text}))[pool.POOL_ASSET]
    assert hashlib.sha256(out.encode()).hexdigest() == pool.TEXT_SHA256
    f = tmp_path / pool.POOL_ASSET
    f.write_text(out)
    pool.verify_text_set(f)                                  # text, schema, shape, canonical (d4315c33...) and payload-list pins
    assert pool.CANONICAL_SHA256 == "d4315c338f44855e1162099c4a42792f0368c60a854f6d047930fa865ef976d8"
    assert len(NP.pack_assets({pool.POOL_ASSET: text})) < len(text) // 5


def test_rehearsal_bootstrap_z_roundtrip_and_pin():
    import rehearsal_artifact as ra
    with pytest.raises(ValueError, match="REPLAY_ARTIFACT_HASH"):
        NP.rehearsal_bootstrap_z(b"not the artifact")
    src = Path(NP.__file__).read_text()
    assert "zlib.decompress(base64.b64decode(embedded_rehearsal_b64))" in src and ra.PINNED_BYTES == 88749


def test_rehearsal2_builder_uses_the_pack_and_the_size_gate():
    src = (ROOT / "tools" / "experiment_packaging" / "build_y72_rehearsal2.py").read_text()
    assert "NP.assets_bootstrap(assets)" in src and "NP.rehearsal_bootstrap_z(" in src
    assert src.index("NP.check_size(len(data))") < src.index('(kernel / "replay.ipynb").write_bytes(data)')
    assert '"assets=" + repr(assets)' not in src
    val = (ROOT / "tools" / "experiment_packaging" / "validate_y72_rehearsal2.py").read_text()
    for case in ("size_gate_package", "negative_oversized_notebook", "negative_tampered_pack_blob", "CHECKPOINT_ARG_MUST_BE_FILE"):
        assert case in val
