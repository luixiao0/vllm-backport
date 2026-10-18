# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Install the MXFP8 development overlay on its source-verified radix base."""

import hashlib
import importlib.util
import json
import os
import py_compile
import shutil
from pathlib import Path

root = Path(importlib.util.find_spec("vllm").origin).parent
staging = Path("/tmp/sm80-mxfp8")
updates = {
    "envs.py": (
        "envs.py",
        "0c824636c5db801a7b355b8707618d7000c7e9c8489d435d3ebad4a9e85c3d18",
    ),
    "model_executor/kernels/linear/__init__.py": (
        "linear_init.py",
        "96efc733c328a85821fceae62645a4d9e64e92cf6db8bf9748b2f05497cfa1c8",
    ),
    "models/deepseek_v4_1/ampere/ampere_sparse.py": (
        "ampere_sparse.py",
        "1ffc1cbefa3351f1a7f755e8820616069a1b12e10520f50f63876dbe566b2456",
    ),
    "model_executor/kernels/linear/mxfp8/sm80.py": ("sm80.py", None),
}
for relative, (_, expected) in updates.items():
    target = root / relative
    if expected is None:
        if target.exists():
            raise RuntimeError(f"Unexpected existing source: {relative}")
    elif hashlib.sha256(target.read_bytes()).hexdigest() != expected:
        raise RuntimeError(f"Unsupported base source: {relative}")

hashes = {}
for relative, (source, _) in updates.items():
    target = root / relative
    shutil.copyfile(staging / source, target)
    py_compile.compile(str(target), doraise=True)
    hashes[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
Path("/opt/sm80-mxfp8.json").write_text(
    json.dumps(
        {
            "files": hashes,
            "source_revision": os.environ.get("SOURCE_REVISION", "unknown"),
        },
        indent=2,
    )
    + "\n"
)
shutil.rmtree(staging)
