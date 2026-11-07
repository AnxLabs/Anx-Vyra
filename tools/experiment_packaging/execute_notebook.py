"""Run one notebook cell file inside an already isolated process.

This file is the sandbox entrypoint. It does not create a sandbox of its own
and it does not submit anything.
"""
import json
import os
import socket
import sys
from pathlib import Path

network_blocked = False
probe = socket.socket()
probe.settimeout(0.2)
try:
    probe.connect(("1.1.1.1", 443))
except OSError:
    network_blocked = True
finally:
    probe.close()

submission_exists = Path("/submission").exists()
Path("/kaggle/working/sandbox_probe.json").write_text(json.dumps({
    "network_blocked": network_blocked,
    "network_interfaces": socket.if_nameindex(),
    "outside_workspace_visible": Path("/workspace").exists(),
    "submission_exists": submission_exists,
    "harness_only": ["/tools", "/runtime", "/usr", "/lib"],
    "runtime_prefix": sys.prefix,
    "python_path_initial": sys.path,
    "credential_env_present": any(k.startswith("KAGGLE_") for k in os.environ),
}, indent=2))
if not network_blocked or Path("/workspace").exists():
    raise RuntimeError("isolation unavailable")
if submission_exists:
    raise RuntimeError("SANDBOX_PRIVILEGED_MOUNT /submission")

nb = json.loads(Path(sys.argv[1]).read_text())
for i, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        code = "".join(cell["source"])
        exec(compile(code, f"notebook_cell_{i}", "exec"), globals())
