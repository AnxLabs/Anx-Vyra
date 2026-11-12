"""Fail-closed resume from checkpoint_300 of the y72fix2 run, for its 300 -> 400 continuation (prereg/y72fix2_cont400.json).

checkpoint_300.pt was written by the y72fix2 run (per-task clip 0.5, replay weight 0.25) at update 300; its SHA-256 and size are pinned
below (they equal the run's own dump record). It is a complete v2 state, so nothing is re-initialized. Nothing about the configuration
changes: replay weight stays exactly the saved 0.25, every other optimizer-config field must equal the saved one, per-task clip 0.5.

Order: SHA-256/size of the one exact path BEFORE torch.load, then schema/update/config/sampler/plan hashes, the live 400-entry plan,
device gate, complete restore (the saved sampler record refers to the 300-entry v1 plan, so restore_full is given plan[0:300] and the
v1 module), then in the notebook the model-state hash (always fatal) and the exact integer counts of the restored model at 300, which
must equal the counts the y72fix2 run recorded at 300 (zero tolerance; fatal on GPU, recorded in the CPU smoke).
No discovery, no retry, no restart.
"""
import continuation as cont
import plan_extension as pe300
import plan_extension400 as pe400
from continuation import ContinuationGateError, _fail  # noqa: F401

CKPT300_SHA256 = "5df9a3f8bcf04dbeac6b15d96a312e3fa3b425aef60639a603ece3c19fea45a3"
CKPT300_BYTES = 197237651
CKPT300_DATASET_ID = "@@KUSER@@/@@SLUG_CKPT300@@"
CKPT300_MOUNT = "@@KROOT@@/input/datasets/@@KUSER@@/@@SLUG_CKPT300@@"
CKPT300_PATH = CKPT300_MOUNT + "/checkpoint_300.pt"
RESUME_UPDATE = 300
FINAL_UPDATE = 400
CHECKPOINT_UPDATES = (325, 350, 375, 400)
SAVED_LAMBDA = 0.25
NEW_LAMBDA = 0.25          # unchanged: this is a continuation of the same configuration
MODEL_STATE_SHA256_300 = "8fecaa8b159d2979b93322b896b9709824ea354a2002c5eafae104cbcaeceb9d"
SAVED_CONFIG_EXCEPT_LAMBDA = {"class": "SliceAdam", "B1": 0.9, "B2": 0.999, "EPS": 1e-08, "LR": 0.0003,
                              "grad_clip_max_norm": 1.0, "replay_batch": 2, "seed": 0}
SAVED_SAMPLER = {"next_update_index": 300, "next_update_number": 301, "replay_rule": "idx=(s*2+j)%40", "replay_next_idx": [0, 1],
                 "plan_length": 300, "plan_prefix200_sha256": pe400.PINNED_PREFIX_PLAN_SHA256,
                 "plan_full_sha256": pe400.PINNED_PREFIX300_PLAN_SHA256, "plan_extension_policy": pe400.BASE_POLICY_VERSION}
# Exact integer counts the y72fix2 run recorded at update 300 (result.json continuation_checkpoints['300']), pinned.
COUNTS_300 = {
    "Y72_unseen": {"tokens_correct": 1558, "tokens": 1600, "segments_exact": 159, "segments": 200},
    "X32_unseen": {"tokens_correct": 2446, "tokens": 2560, "segments_exact": 229, "segments": 320},
    "X64_unseen": {"tokens_correct": 1665, "tokens": 1920, "segments_exact": 44, "segments": 240},
    "X109_unseen": {"tokens_correct": 948, "tokens": 960, "segments_exact": 109, "segments": 120},
    "Y72v2_600": {"tokens_correct": 4694, "tokens": 4800, "segments_exact": 496, "segments": 600},
}
X64_300 = {
    "train": {"tokens_correct": 1816, "tokens": 1920, "segments_exact": 139, "segments": 240},
    "heldout": {"tokens_correct": 1665, "tokens": 1920, "segments_exact": 44, "segments": 240},
}


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
    if new_lambda != SAVED_LAMBDA:
        _fail("LAMBDA_CHANGED", f"{new_lambda} != saved {SAVED_LAMBDA}: the continuation keeps the configuration")
    if cuda_required and (ck["cuda_rng"] is None or len(ck["cuda_rng"]) != cont.REQUIRED_CUDA_DEVICES):
        _fail("CKPT_SCHEMA", "cuda_rng must hold the two saved device states")
    if ck["sampler"] != SAVED_SAMPLER:
        _fail("SAMPLER_MISMATCH", "saved sampler record differs from the pinned update-300 record")


def check_live_plan(plan):
    if len(plan) != pe400.FULL_LEN:
        _fail("PLAN_PREFIX_MISMATCH", f"live plan length {len(plan)} != {pe400.FULL_LEN}")
    cont.check_plan_prefix(plan, plan_module=pe400)
    if pe400.plan_sha256(plan[:RESUME_UPDATE]) != pe400.PINNED_PREFIX300_PLAN_SHA256:
        _fail("PLAN_PREFIX_MISMATCH", "live plan[0:300] differs from the plan that produced checkpoint_300")
    if pe400.plan_sha256(plan) != pe400.PINNED_FULL_PLAN_SHA256:
        _fail("PLAN_PREFIX_MISMATCH", "live 400-entry plan hash differs from the pinned plan")


def load_checkpoint300(path, *, model, opt, plan, lib, device, replay_batch=2, cpu_smoke=False,
                       expected_sha=CKPT300_SHA256, expected_bytes=CKPT300_BYTES, torch_mod=None):
    if torch_mod is None:
        import torch as torch_mod
    raw = cont.read_verified_bytes(path, expected_sha, expected_bytes)                    # 1. BEFORE torch.load
    ck = cont.load_full_checkpoint(raw)                                                   # 2. weights_only load of verified bytes
    validate_v2_resume(ck, cuda_required=not cpu_smoke)                                   # 3. schema/update/config/sampler
    check_live_plan(plan)                                                                 # 4. live 400-entry plan
    dev = cont.check_device(model, ck, device=device, cpu_smoke=cpu_smoke, torch_mod=torch_mod)   # 5. two T4, cuda:0
    rec = cont.restore_full(model, opt, ck, lib=lib, plan=plan[:RESUME_UPDATE], plan_module=pe300, replay_batch=replay_batch,
                            cuda=dev["cuda_rng_restored"])                                # 6. everything, exactly
    return {"path": str(path), "sha256": expected_sha, "bytes": expected_bytes, "update": ck["update"], "device": dev,
            "restored_components": rec["components"], "saved_lambda": SAVED_LAMBDA, "new_lambda": NEW_LAMBDA,
            "saved_config_except_lambda": SAVED_CONFIG_EXCEPT_LAMBDA, "provenance_of_checkpoint": ck["provenance"],
            "plan_policy": pe400.POLICY_VERSION, "plan_full_sha256": pe400.PINNED_FULL_PLAN_SHA256}


def check_resume_restoration(model, x64, counts, *, strict, x64_ck):
    """Parameter hash always fatal; the integer counts must equal COUNTS_300 / X64_300 exactly (fatal when strict)."""
    sha = x64_ck._state_sha(model)
    rec = {"model_state_sha256": sha, "model_state_matches": sha == MODEL_STATE_SHA256_300}
    if not rec["model_state_matches"]:
        _fail("RESTORE_CHECK", f"model_state_sha256 {sha} != {MODEL_STATE_SHA256_300}")
    keys = ("tokens_correct", "tokens", "segments_exact", "segments")
    x64_diff = {k: {f: [x64[k][f], X64_300[k][f]] for f in keys if x64[k][f] != X64_300[k][f]} for k in ("train", "heldout")}
    x64_diff = {k: v for k, v in x64_diff.items() if v}
    cnt_diff = {}
    for name, want in COUNTS_300.items():
        got = counts.get(name)
        d = {f: [None if got is None else got[f], want[f]] for f in keys if got is None or got[f] != want[f]}
        if d:
            cnt_diff[name] = d
    rec.update({"x64_counts_match": not x64_diff, "x64_counts_diff": x64_diff, "eval_counts_match": not cnt_diff,
                "eval_counts_diff": cnt_diff, "tolerance": 0})
    if strict and (x64_diff or cnt_diff):
        _fail("RESTORE_CHECK", f"restored checkpoint_300 does not reproduce the recorded integer counts: x64_diff={x64_diff} eval_diff={cnt_diff}")
    return rec
