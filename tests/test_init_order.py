"""Use-before-binding checks for the packaged treatment.

The expanded40 Kaggle run (06dd71e) died with NameError: preclip_diag at
treatment.py:149. The baseline readout used the helper six lines before the
loader bound it. CPU preflight exits before that line, so nothing caught it.
"""
import ast
import json
import os
import sys
from pathlib import Path

import pytest

PACKAGING = Path(__file__).resolve().parents[1] / "tools" / "experiment_packaging"
sys.path.insert(0, str(PACKAGING))

from build import prepare_treatment  # noqa: E402
from name_order import used_before_binding  # noqa: E402
from test_packaging_fail_closed import MINIMAL_TREATMENT  # noqa: E402

# Bound by exec() of the pinned corpus builder at treatment.py:55.
EXEC_BOUND = {"build"}


def test_preclip_diag_is_bound_before_its_first_use():
    patched, _builder = prepare_treatment(MINIMAL_TREATMENT, "builder-text")
    # The fixture's line order is not the real flow, so anchor on the
    # baseline readout: in the real treatment it is the first use.
    assert patched.count("preclip_diag=_dmod") == 1
    assert patched.index("preclip_diag=_dmod") < patched.index("R['baseline_rehearsal']=preclip_diag.rehearsal_readout")


def test_checker_flags_use_before_import():
    bad = "x = mod.f()\nimport mod\n"
    assert used_before_binding(bad) == [("mod", 1)]
    good = "import mod\nx = mod.f()\ndef g():\n    return later\nlater = 1\n"
    assert used_before_binding(good) == []
    comp = "ps = [p for _, _, ps in rows for p in ps]\n"
    assert used_before_binding(comp, predefined={"rows"}) == []


def _assets(notebook):
    code = "".join(json.loads(Path(notebook).read_text())["cells"][0]["source"])
    node = next(
        n for n in ast.parse(code).body
        if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", None) == "assets"
    )
    return code, ast.literal_eval(node.value)


@pytest.mark.skipif(not os.environ.get("VYRA_PACKAGED_NOTEBOOK"), reason="set VYRA_PACKAGED_NOTEBOOK to a built replay.ipynb")
def test_packaged_notebook_has_no_use_before_binding():
    code, assets = _assets(os.environ["VYRA_PACKAGED_NOTEBOOK"])
    found = {"bootstrap": used_before_binding(code)}
    for name, text in assets.items():
        if name.endswith(".py"):
            extra = EXEC_BOUND if name == "treatment.py" else ()
            found[name] = used_before_binding(text, predefined=extra)
    assert {k: v for k, v in found.items() if v} == {}
