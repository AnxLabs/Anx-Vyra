"""The private Kaggle config: fail-closed loading, token expansion, and builders that refuse to run without it."""
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PKG = ROOT / "tools" / "experiment_packaging"
sys.path.insert(0, str(PKG))

import kaggle_config as K  # noqa: E402

BUILDERS = ("build.py", "build_continuation.py", "build_y72fix.py", "build_y72fix2.py", "build_y72fix2_cont400.py")


def _cfg(**over):
    cfg = json.loads((ROOT / "kaggle_config.example.json").read_text())
    cfg.update(over)
    return cfg


def _argv(builder, tmp_path, out):
    names = ["source-notebook", "x-archive", "y-archive", "builder-source", "archive-audit", "replay-artifact", "output"]
    if builder != "build.py":
        names.insert(-1, "checkpoint")
    args = [sys.executable, str(PKG / builder)]
    for name in names:
        args += ["--" + name, str(out if name == "output" else tmp_path / "unused")]
    return args


@pytest.fixture
def use_config(tmp_path, monkeypatch):
    def apply(cfg, name="cfg.json", raw=None):
        path = tmp_path / name
        path.write_text(raw if raw is not None else json.dumps(cfg))
        monkeypatch.setenv(K.ENV_PATH, str(path))
        K._CACHE.clear()
        return path
    yield apply
    K._CACHE.clear()


def test_missing_config_fails_closed_with_a_clear_error(monkeypatch, tmp_path):
    monkeypatch.setenv(K.ENV_PATH, str(tmp_path / "does_not_exist.json"))
    K._CACHE.clear()
    with pytest.raises(K.KaggleConfigError, match="kaggle config missing"):
        K.load()
    with pytest.raises(K.KaggleConfigError, match="kaggle config missing"):
        K.expand("@@KUSER@@/x")
    K._CACHE.clear()


def test_text_without_tokens_never_needs_the_config(monkeypatch, tmp_path):
    monkeypatch.setenv(K.ENV_PATH, str(tmp_path / "does_not_exist.json"))
    K._CACHE.clear()
    assert K.expand("plain text, no tokens") == "plain text, no tokens"
    K._CACHE.clear()


@pytest.mark.parametrize("builder", BUILDERS)
def test_every_builder_refuses_to_start_without_the_config(builder, tmp_path):
    out = tmp_path / "out"
    env = dict(os.environ, **{K.ENV_PATH: str(tmp_path / "does_not_exist.json")})
    env.pop(K.ENV_ALLOW_EXAMPLE, None)
    args = _argv(builder, tmp_path, out)
    proc = subprocess.run(args, capture_output=True, text=True, env=env, timeout=120, cwd=tmp_path)
    assert proc.returncode != 0
    assert "kaggle config missing" in proc.stderr
    assert not out.exists()                                   # nothing was built or written


@pytest.mark.parametrize("builder", BUILDERS)
def test_builders_refuse_the_placeholder_template(builder, tmp_path):
    out = tmp_path / "out"
    env = dict(os.environ, **{K.ENV_PATH: str(ROOT / "kaggle_config.example.json")})
    env.pop(K.ENV_ALLOW_EXAMPLE, None)
    args = _argv(builder, tmp_path, out)
    proc = subprocess.run(args, capture_output=True, text=True, env=env, timeout=120, cwd=tmp_path)
    assert proc.returncode != 0 and "placeholder template" in proc.stderr
    assert not out.exists()


def test_malformed_or_incomplete_config_fails_closed(use_config):
    use_config(None, raw="{not json")
    with pytest.raises(K.KaggleConfigError, match="not valid JSON"):
        K.load()
    use_config(_cfg(username=""))
    with pytest.raises(K.KaggleConfigError, match="username"):
        K.load()
    slugs = dict(_cfg()["slugs"])
    slugs.pop("ckpt225")
    use_config(_cfg(slugs=slugs))
    with pytest.raises(K.KaggleConfigError, match="slugs.ckpt225"):
        K.load()
    use_config(_cfg(root="@@KROOT@@"))
    with pytest.raises(K.KaggleConfigError, match="must not contain"):
        K.load()


def test_unknown_and_leftover_tokens_fail_closed(use_config):
    use_config(_cfg())
    with pytest.raises(K.KaggleConfigError, match="unknown kaggle config token"):
        K.expand("a @@NO_SUCH_TOKEN@@ b")
    with pytest.raises(K.KaggleConfigError, match="unexpanded"):
        K.assert_clean("left over @@KUSER@@ here", "unit")
    K.assert_clean(K.expand("fine @@KUSER@@"), "unit")


def test_expansion_is_config_driven_and_idempotent(use_config):
    use_config(_cfg(username="user-one", root="/root-one"))
    one = K.expand("@@KROOT@@/input/datasets/@@KUSER@@/@@SLUG_X_BUNDLE@@")
    assert one == "/root-one/input/" + "data" + "sets/user-one/example-x-dataset"
    assert K.expand(one) == one
    assert K.expand_value({"k": ["@@KUSER@@", ("@@KROOT@@",)]}) == {"k": ["user-one", ("/root-one",)]}
    assert K.mount("x_bundle") == one and K.dataset_id("ckpt225") == "user-one/example-ckpt225-dataset"
    use_config(_cfg(username="user-two", root="/root-two"), name="two.json")
    assert K.expand("@@KUSER@@") == "user-two"


def test_expand_module_expands_nested_constants(use_config):
    use_config(_cfg(username="user-one"))
    ns = {"A": "@@KUSER@@", "B": (("@@KUSER@@", 1),), "C": 7, "__doc__": "@@KUSER@@"}
    K.expand_module(ns)
    assert ns["A"] == "user-one" and ns["B"] == (("user-one", 1),) and ns["C"] == 7 and ns["__doc__"] == "@@KUSER@@"


def test_asset_modules_carry_tokens_and_builders_expand_them(use_config):
    use_config(_cfg(username="user-one", root="/root-one"))
    raw = (PKG / "continuation.py").read_text()
    assert "@@KUSER@@" in raw and "user-one" not in raw
    expanded = K.read_asset(PKG / "continuation.py")
    assert "user-one" in expanded and "@@" not in expanded
    gate = K.read_asset(PKG / "dependency_gate.py")
    assert "/root-one/working/dependency_lock.json" in gate


def test_example_template_is_placeholders_only():
    cfg = json.loads((ROOT / "kaggle_config.example.json").read_text())
    assert cfg["example"] is True
    assert set(cfg["slugs"]) == set(K.REQUIRED_SLUGS) | set(K.OPTIONAL_SLUGS)
    real = ROOT / "kaggle_config.json"
    if real.is_file():
        real_cfg = json.loads(real.read_text())
        assert not real_cfg.get("example")
        values = [real_cfg["username"], real_cfg["root"], real_cfg["local_inputs"], *real_cfg["slugs"].values()]
        assert not any(v in (ROOT / "kaggle_config.example.json").read_text() for v in values)


def test_optional_slugs_are_optional_but_fail_closed_when_used(use_config):
    slugs = {k: v for k, v in _cfg()["slugs"].items() if k not in K.OPTIONAL_SLUGS}
    use_config(_cfg(slugs=slugs))
    assert K.expand("@@SLUG_X_BUNDLE@@") == "example-x-dataset"              # older configs still load
    with pytest.raises(K.KaggleConfigError, match="unknown kaggle config token"):
        K.expand("@@SLUG_CKPT300@@")
    use_config(_cfg(slugs=dict(slugs, ckpt300="")), name="empty.json")
    with pytest.raises(K.KaggleConfigError, match="slugs.ckpt300"):
        K.load()
    use_config(_cfg(), name="full.json")
    assert K.dataset_id("ckpt300") == "example-user/example-ckpt300-dataset"
    assert K.expand("@@SLUG_KERNEL_Y72FIX2_CONT400@@") == "example-kernel-y72fix2-cont400"
