"""Three isolated rejections. They require a real built package and real inputs.

Nothing here submits, and a failure path does not import the model. The
expected messages come from the generated notebook, not from a substitute.
"""
import argparse
import ast
import json
import shutil
import time
from pathlib import Path

from validate_package import validate
import kaggle_config as K


CASES = {
    "missing_dependency": ("BOOTSTRAP_ASSET_SET_MISMATCH", "missing"),
    "corrupt_hash": ("BOOTSTRAP_CORRUPT entry.py", "corrupt"),
    "wrong_path": (K.expand("missing dataset root @@KUSER@@/@@SLUG_X_BUNDLE@@"), "path"),
}


def stage_negative_kernel(package, target, notebook_text):
    """Copy kernel companions, then write the mutated notebook.

    Companions are not Kaggle inputs. The rehearsal bytes travel inside the
    notebook. The wrong-path case still has to fail on the dataset root.
    """
    source = Path(package) / "kernel"
    dest = Path(target) / "kernel"
    dest.mkdir(parents=True, exist_ok=True)
    for item in sorted(source.iterdir()):
        if item.name == "replay.ipynb":
            continue
        if not item.is_file():
            raise ValueError("NEGATIVE_FIXTURE_UNEXPECTED " + item.name)
        shutil.copy2(item, dest / item.name)
    (dest / "replay.ipynb").write_text(notebook_text)
    missing = [
        item.name for item in source.iterdir()
        if item.name != "replay.ipynb" and not (dest / item.name).is_file()
    ]
    if missing:
        raise ValueError("NEGATIVE_FIXTURE_INCOMPLETE " + ",".join(missing))
    return dest


def run(package, inputs, work_root, logs, runtime):
    package = Path(package)
    inputs = Path(inputs)
    work_root = Path(work_root)
    logs = Path(logs)
    logs.mkdir(parents=True, exist_ok=True)
    original = json.loads((package / "kernel" / "replay.ipynb").read_text())
    code = "".join(original["cells"][0]["source"])
    cases = CASES
    results = {}
    start = time.monotonic()
    for name, (message, kind) in cases.items():
        tree = ast.parse(code)
        if kind in ("missing", "corrupt"):
            assignment = next(
                n for n in tree.body
                if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "assets" for t in n.targets)
            )
            assets = ast.literal_eval(assignment.value)
            if kind == "missing":
                del assets["make_corpus_X.py"]
            else:
                assets["entry.py"] += "\n# intentional corrupt fixture\n"
            assignment.value = ast.parse(repr(assets), mode="eval").body
        else:
            mapping = next(
                n for n in tree.body
                if isinstance(n, ast.Assign)
                and any(isinstance(t, ast.Name) and t.id == "mapping" for t in n.targets)
            )
            for i, key in enumerate(mapping.value.keys):
                if isinstance(key, ast.Name) and key.id == "XID":
                    mapping.value.values[i] = ast.Constant(
                        K.expand("@@KROOT@@/input/datasets/@@KUSER@@/WRONG-X-PATH")
                    )
        target = work_root / name
        mutated = json.loads(json.dumps(original))
        mutated["cells"][0]["source"] = [ast.unparse(ast.fix_missing_locations(tree)) + "\n"]
        stage_negative_kernel(package, target, json.dumps(mutated))
        log = logs / ("negative_" + name + ".log")
        try:
            validate(target, inputs, target / "work", log, runtime)
        except RuntimeError:
            text = log.read_text()
            if message not in text:
                raise AssertionError("Wrong failure for " + name + ": " + text[-2000:])
            results[name] = {
                "status": "EXPECTED_REJECT",
                "message": message,
                "log": str(log),
            }
        else:
            raise AssertionError("Unexpectedly accepted " + name)
        print("PASS_EXPECTED_REJECTION", name, message)
    report = {"status": "PASS", "elapsed_s": time.monotonic() - start, "tests": results}
    (work_root / "NEGATIVE_REPORT.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", required=True)
    parser.add_argument("--inputs", required=True)
    parser.add_argument("--work-root", required=True)
    parser.add_argument("--logs", required=True)
    parser.add_argument("--runtime", required=True)
    args = parser.parse_args()
    run(args.package, args.inputs, args.work_root, args.logs, args.runtime)
