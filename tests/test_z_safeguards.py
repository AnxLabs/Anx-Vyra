"""Tests for the Experiment Z safety mechanisms added on forge/z-builder.

These cover guards that exist because each one failed open at least once during this build:
a check that passes because it verified nothing, a verdict that cannot reach PASS, and a
configuration that reports itself as an arm it is not. A guard that cannot fail is not a guard,
so every test here asserts the refusal path, not just the happy path.
"""
import ast
import pathlib
import subprocess
import sys

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
YTRAIN = ROOT / "experiments" / "y" / "y_train.py"
BUILDER = ROOT / "experiments" / "y" / "make_corpus_Z.py"


def _load_helpers(src_path, names, extra_ns):
    """Compile only the named top-level functions out of a trainer/builder and return its namespace.

    The trainer is a script, not a module: importing it would start a training run. Parsing out the
    pure helpers lets the real implementations be tested rather than a copy that can drift from them.
    """
    tree = ast.parse(src_path.read_text())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    missing = names - {n.name for n in keep}
    assert not missing, f"{src_path.name} no longer defines {sorted(missing)}"
    ns = dict(extra_ns)
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(src_path), "exec"), ns)
    return ns


# ---- the verdict must be able to say PASS ----------------------------------------------------

def _verdict_ns(gaps, geom, targeted=(40, 72)):
    return _load_helpers(
        YTRAIN,
        {"last", "_targeted_key", "_targeted_row", "_targeted_selftest", "_selftest_keys"},
        {"R": {}, "GAPS": list(gaps), "GEOM": list(geom), "_TARGETED": list(targeted),
         "ARM_SPEC": {"jitter_3e4": ("jitter", 3e-4)}, "STEPS": 600})


def test_verdict_can_reach_pass_under_approved_arm_a():
    """Arm A trains 28..36/60..68/105..113, so 40 and 72 are gap probes, never trained.

    The published lookup read g{d}_train, which the table only emits for trained geometries, so both
    lookups were None and a flawless run reported 'the collapsed distances did not recover'.
    """
    ns = _verdict_ns([40, 44, 48, 52, 72, 76, 80, 84, 88, 92, 96, 121, 128],
                     [28, 32, 36, 60, 64, 68, 105, 109, 113])
    out = ns["_targeted_selftest"]()
    assert out["all_recovered"] is True
    assert {r["source_namespace"] for r in out["rows"]} == {"gap"}, "40/72 must read as gap probes"
    assert all(r["was_trained"] is False for r in out["rows"]), \
        "a never-trained distance read as trained would be a manufactured number"


def test_verdict_reads_trained_namespace_when_that_is_where_the_distance_lives():
    """Y's own arm trained 40 and 72 as +/-8 offsets, so g40_train existed and was the right key."""
    ns = _verdict_ns([44, 48, 52, 76, 80, 84, 88, 92, 96, 121, 128],
                     [24, 28, 32, 36, 40, 56, 60, 64, 68, 72, 101, 105, 109, 113, 117])
    out = ns["_targeted_selftest"]()
    assert out["all_recovered"] is True
    assert {r["source_namespace"] for r in out["rows"]} == {"trained_geometry"}
    assert all(r["was_trained"] is True for r in out["rows"])


def test_verdict_refuses_to_start_when_the_targeted_distance_exists_in_neither_namespace():
    """A run whose own verdict cannot reach PASS is a broken instrument, not a negative result."""
    ns = _verdict_ns([44, 48, 52, 76, 80, 84, 88, 92, 96, 121, 128],
                     [28, 32, 36, 60, 64, 68, 105, 109, 113])
    with pytest.raises(SystemExit) as exc:
        ns["_targeted_selftest"]()
    assert "PASS_UNREACHABLE_SELFTEST" in str(exc.value)


def test_a_distance_missing_from_both_namespaces_is_recorded_not_invented():
    ns = _verdict_ns([44, 48], [28, 32], targeted=(40,))
    key, trained, source = ns["_targeted_key"](40)
    assert (key, trained, source) == (None, None, "absent")


# ---- the builder must refuse to build an arm it was not asked for ----------------------------

def _builder_env(arm=None, **over):
    import os
    env = {k: v for k, v in os.environ.items()
           if not k.startswith(("Z_", "Y_", "T2_", "VYRA_"))}
    # Paths come from the environment so this test carries no machine-specific location into a
    # public repo. Point Z_TEST_SRC at a bundle holding corpus.pt/probe_Q.pt to run it; without one
    # the builder-refusal tests skip, since they need real artifacts to reach the checks.
    src = os.environ.get("Z_TEST_SRC")
    if not src:
        pytest.skip("Z_TEST_SRC not set; builder refusal tests need a real artifact bundle")
    import tempfile
    tmp = tempfile.mkdtemp(prefix="ztest_")
    env.update({"PYTHONPATH": os.environ.get("Z_TEST_PYTHONPATH", ""),
                "T2_SRC": src, "VYRA_LAB": os.path.join(tmp, "lab"),
                "Z_OUT": os.path.join(tmp, "out")})
    if arm:
        env["Z_ARM"] = arm
    env.update(over)
    return env


@pytest.mark.parametrize("over,needle", [
    ({}, "Z_ARM is required"),
    ({"Z_ARM": "A", "Z_N_TR": "8"}, "not that arm"),
    ({"Z_ARM": "A", "Z_JITTER": "-8,-4,0,4,8"}, "GAP_BAND_COLLISION"),
])
def test_builder_refuses_misconfigured_arms(over, needle):
    """Every one of these produced a corpus that reported itself as Arm A while not being it."""
    env = _builder_env(**over)
    p = subprocess.run([sys.executable, str(BUILDER)],
                       env=env, capture_output=True, text=True, timeout=300)
    assert p.returncode != 0, f"builder accepted a misconfigured arm: {over}"
    assert needle in (p.stdout + p.stderr), (p.stdout + p.stderr)[-800:]


# ---- the trainer must hold its safeguards in the module that loaded --------------------------

def test_trainer_self_audit_is_present_and_checks_the_loaded_module():
    src = YTRAIN.read_text()
    tree = ast.parse(src)
    names = {n.name for n in tree.body if isinstance(n, ast.FunctionDef)}
    assert "_self_audit" in names, "the loaded-module audit is gone"
    for required in ("EVAL_BUDGET_MODE", "RULE5_PASSES", "_resolve_steps",
                     "NON_SKIPPABLE_BLOCKS"):
        assert required in src, f"{required} is checked at runtime but no longer defined"
    # the audit has to write what it verified into the artifact, not just pass silently
    assert '"safeguard_audit"' in src


def test_rule5_dormancy_is_fatal_not_a_printed_note():
    """Dormancy is fail-open: the run finishes and no RULE 5 record reads as learning."""
    src = YTRAIN.read_text()
    assert "RULE5_UNREACHABLE" in src
    assert "RULE5_DORMANT" not in src, "the print-and-train-on path is back"


def test_empty_t2_expect_is_fatal():
    """An unset hash expectation made the preflight gate pass having hashed nothing."""
    src = YTRAIN.read_text()
    assert "T2_EXPECT_MISSING" in src
    assert 'os.environ.get("T2_EXPECT", "{}")' not in src, \
        "the vacuous default is back: an empty T2_EXPECT must not silently disable the gate"


def test_checkpoint_identity_is_pinned_by_file_hash():
    """experiment/step come from inside the file; only a file hash ties the run to those bytes."""
    src = YTRAIN.read_text()
    assert "T2_CKPT_SHA256_MISMATCH" in src
    assert "T2_CKPT_SHA256_MISSING" in src


def test_ambiguous_input_basenames_are_refused():
    src = YTRAIN.read_text()
    assert "AMBIGUOUS_INPUT" in src


def test_gap_mean_stays_over_the_original_eleven():
    """gap_mean is a pre-registered verdict input; all eleven Y gaps are 0.0 at baseline."""
    src = YTRAIN.read_text()
    assert "GAP_MEAN_DISTANCES" in src
    assert "gap_mean_extended" in src
    # the extended mean is for information and must never enter the verdict expression
    verdict = src[src.index('gap_ok = all('):]
    assert "gap_mean_extended" not in verdict[:400]


def test_metric_gate_fails_on_a_metric_it_cannot_resolve():
    """The reference carries the pre-rename names, so unaliased lookups silently vanished."""
    src = YTRAIN.read_text()
    assert "METRIC_GATE_UNRESOLVED" in src
    assert "_ALIASES" in src


def test_gap_mean_is_carried_on_a_measurement_that_can_move():
    """record_exact is ~token_acc**8 and pins at 0.000 below about 0.63 token accuracy.

    Y's eleven gaps were all exactly 0.0 for that reason while their token accuracy ranged
    0.09-0.56, so a +0.02 improvement test on the record-exact mean could only ever fail.
    """
    src = YTRAIN.read_text()
    assert 'out["gap_mean_metric"] = "token_acc"' in src
    assert "gap_mean_record_exact" in src, "the pre-registered figure must still be recorded"
    block = src[src.index('_mean_set = [g for g in GAPS'):]
    block = block[:block.index('out["gap_mean_extended"]')]
    assert "token_acc" in block, "gap_mean must average token accuracy, not record_exact"
    assert 'out["gap_mean_record_exact"]' in block, \
        "record_exact must survive beside it, since that is the pre-registered figure"
