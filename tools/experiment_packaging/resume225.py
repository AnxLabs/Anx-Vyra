"""Fail-closed resume from checkpoint_225 (schema v2) of the 200 -> 300 run, for the replay-weight experiment.

checkpoint_225.pt was written by the 300 run (sha256 pinned below, taken from the downloaded run outputs). It holds the
complete state, so unlike checkpoint_200 nothing is re-initialized: parameters, SliceAdam state and expert clocks,
torch CPU + both CUDA RNG states, Python and NumPy RNG, router metadata, model mode and sampler position are all restored.
The ONLY intended difference from the saved configuration is replay_lambda 0.25 -> 0.5, which is checked explicitly:
every other optimizer-config field must equal the saved one and the saved lambda must be exactly 0.25.

Order (same discipline as continuation.load_checkpoint200): SHA-256/size of the one exact path BEFORE torch.load, then
schema/update/config/sampler/plan hashes, device gate, restore, model-state hash, then (in the notebook) the exact integer
eval counts of the restored model, which must equal the counts the 300 run recorded at update 225 (zero tolerance).
No discovery, no retry, no restart.
"""
import io

import torch

import continuation as cont
from continuation import ContinuationGateError, _fail

CKPT225_SHA256 = "a84d35bee3323cf60d0f0a09f0eece5ffb4c1e7bddbd196d4c478df15cb1b283"
CKPT225_BYTES = 197237331
CKPT225_DATASET_ID = "@@KUSER@@/@@SLUG_CKPT225@@"
CKPT225_MOUNT = "@@KROOT@@/input/datasets/@@KUSER@@/@@SLUG_CKPT225@@"
CKPT225_PATH = CKPT225_MOUNT + "/checkpoint_225.pt"
RESUME_UPDATE = 225
SAVED_LAMBDA = 0.25
NEW_LAMBDA = 0.5
# sha256 over name+bytes of the state dict (x64_checkpoint._state_sha), computed from the downloaded checkpoint_225.pt
MODEL_STATE_SHA256_225 = "0990d7c25ec3108bd05f214e1cbfb5495cf12cfceec68e9fddfd7f404bc61482"
SAVED_CONFIG_EXCEPT_LAMBDA = {"class": "SliceAdam", "B1": 0.9, "B2": 0.999, "EPS": 1e-08, "LR": 0.0003,
                              "grad_clip_max_norm": 1.0, "replay_batch": 2, "seed": 0}
# Exact integer counts recomputed by the 300 run at update 225 (result.json continuation_checkpoints['225']), pinned.
COUNTS_225 = {
    "Y72_unseen": {"tokens_correct": 1565, "tokens": 1600, "segments_exact": 168, "segments": 200},
    "X32_unseen": {"tokens_correct": 2336, "tokens": 2560, "segments_exact": 160, "segments": 320},
    "X64_unseen": {"tokens_correct": 1418, "tokens": 1920, "segments_exact": 11, "segments": 240},
    "X109_unseen": {"tokens_correct": 934, "tokens": 960, "segments_exact": 102, "segments": 120},
}
X64_225 = {
    "train": {"tokens_correct": 1562, "tokens": 1920, "segments_exact": 28, "segments": 240},
    "heldout": {"tokens_correct": 1418, "tokens": 1920, "segments_exact": 11, "segments": 240},
}
PINNED_PREFIX_PLAN_SHA256 = "c3ac1155e0e21b25133d7e76aa63589a98e53d3e80ee3b0c645daa3fcd365ae7"
PINNED_FULL_PLAN_SHA256 = "2c59794e261a88216e5d77141b3373073c6e518545dce8a2578f279e95de86aa"


def validate_v2_resume(ck, *, new_lambda=NEW_LAMBDA, expected_update=RESUME_UPDATE, cuda_required=True):
    if ck.get("schema") != cont.V2_SCHEMA:
        _fail("CKPT_SCHEMA", f"schema {ck.get('schema')!r}")
    missing = [k for k in cont.V2_REQUIRED if k not in ck]
    if missing:
        _fail("CKPT_SCHEMA", f"missing {missing}")
    if ck["update"] != expected_update:
        _fail("CKPT_UPDATE", f"{ck['update']} != {expected_update}")
    cfg = dict(ck["optimizer_config"])
    if cfg.pop("replay_lambda", None) != SAVED_LAMBDA:
        _fail("OPTIMIZER_CONFIG_MISMATCH", f"saved replay_lambda != {SAVED_LAMBDA}")
    if cfg != SAVED_CONFIG_EXCEPT_LAMBDA:
        _fail("OPTIMIZER_CONFIG_MISMATCH", f"{cfg} != {SAVED_CONFIG_EXCEPT_LAMBDA}")
    if new_lambda != NEW_LAMBDA:
        _fail("LAMBDA_NOT_PREREGISTERED", f"{new_lambda} != {NEW_LAMBDA}")
    if cuda_required and (ck["cuda_rng"] is None or len(ck["cuda_rng"]) != cont.REQUIRED_CUDA_DEVICES):
        _fail("CKPT_SCHEMA", "cuda_rng must hold the two saved device states")
    if ck["sampler"].get("plan_prefix200_sha256") != PINNED_PREFIX_PLAN_SHA256 or ck["sampler"].get("plan_full_sha256") != PINNED_FULL_PLAN_SHA256:
        _fail("PLAN_PREFIX_MISMATCH", "saved sampler plan hashes differ from the pinned plan")
    if ck["sampler"].get("next_update_index") != expected_update:
        _fail("SAMPLER_MISMATCH", "next_update_index")


def load_checkpoint225(path, *, model, opt, plan, plan_module, lib, device, replay_batch=2, cpu_smoke=False,
                       expected_sha=CKPT225_SHA256, expected_bytes=CKPT225_BYTES, torch_mod=torch):
    raw = cont.read_verified_bytes(path, expected_sha, expected_bytes)                    # 1. BEFORE torch.load
    ck = cont.load_full_checkpoint(raw)                                                   # 2. weights_only load of verified bytes
    validate_v2_resume(ck, cuda_required=not cpu_smoke)                                   # 3. schema/update/config/plan/sampler
    cont.check_plan_prefix(plan, plan_module=plan_module)                                 # 4. plan prefix of the live plan
    if plan_module.plan_sha256(plan) != PINNED_FULL_PLAN_SHA256:
        _fail("PLAN_PREFIX_MISMATCH", "live full plan hash differs from the pinned full plan")
    dev = cont.check_device(model, ck, device=device, cpu_smoke=cpu_smoke, torch_mod=torch_mod)   # 5. two T4, cuda:0
    rec = cont.restore_full(model, opt, ck, lib=lib, plan=plan, plan_module=plan_module, replay_batch=replay_batch,
                            cuda=dev["cuda_rng_restored"])                                # 6. everything, exactly
    return {"path": str(path), "sha256": expected_sha, "bytes": expected_bytes, "update": ck["update"], "device": dev,
            "restored_components": rec["components"], "saved_lambda": SAVED_LAMBDA, "new_lambda": NEW_LAMBDA,
            "saved_config_except_lambda": SAVED_CONFIG_EXCEPT_LAMBDA, "provenance_of_checkpoint": ck["provenance"]}


def check_resume_restoration(model, x64, counts, *, strict, x64_ck):
    """Parameter hash always fatal; the integer counts must equal COUNTS_225 / X64_225 exactly (fatal when strict)."""
    sha = x64_ck._state_sha(model)
    rec = {"model_state_sha256": sha, "model_state_matches": sha == MODEL_STATE_SHA256_225}
    if not rec["model_state_matches"]:
        _fail("RESTORE_CHECK", f"model_state_sha256 {sha} != {MODEL_STATE_SHA256_225}")
    keys = ("tokens_correct", "tokens", "segments_exact", "segments")
    x64_diff = {k: {f: [x64[k][f], X64_225[k][f]] for f in keys if x64[k][f] != X64_225[k][f]} for k in ("train", "heldout")}
    x64_diff = {k: v for k, v in x64_diff.items() if v}
    cnt_diff = {}
    for name, want in COUNTS_225.items():
        got = counts.get(name)
        d = {f: [None if got is None else got[f], want[f]] for f in keys if got is None or got[f] != want[f]}
        if d:
            cnt_diff[name] = d
    rec.update({"x64_counts_match": not x64_diff, "x64_counts_diff": x64_diff, "eval_counts_match": not cnt_diff,
                "eval_counts_diff": cnt_diff, "tolerance": 0})
    if strict and (x64_diff or cnt_diff):
        _fail("RESTORE_CHECK", f"restored checkpoint_225 does not reproduce the recorded integer counts: x64_diff={x64_diff} eval_diff={cnt_diff}")
    return rec
