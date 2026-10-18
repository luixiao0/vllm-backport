# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Install the development overlay only on its verified base sources."""

import hashlib
import importlib.metadata as md
import importlib.util
import json
import os
import py_compile
import shutil
from pathlib import Path

root = Path(importlib.util.find_spec("vllm").origin).parent
staging = Path("/tmp/sm80-candidate")
updates = {
    "envs.py": (
        "envs.py",
        "eafe500bb45c7327399673553537353ce3b9c22badf07cf07c7e258eb88272d2",
    ),
    "model_executor/kernels/attention/dsa/candidate_blocks.py": (
        "candidate_blocks.py",
        "0d53e4e6a60a37995713bf792089d645b07db1af3e55e36dcc49542705a5db9c",
    ),
    "model_executor/kernels/attention/dsa/candidate_radix.py": (
        "candidate_radix.py",
        None,
    ),
}
for relative, (_, expected) in updates.items():
    target = root / relative
    if expected is None:
        if target.exists():
            raise RuntimeError(f"Unexpected existing source: {relative}")
    else:
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Unsupported base source for {relative}: {actual}")

hashes = {}
for relative, (source, _) in updates.items():
    target = root / relative
    shutil.copyfile(staging / source, target)
    py_compile.compile(str(target), doraise=True)
    hashes[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
Path("/opt/sm80-candidate.json").write_text(
    json.dumps(
        {
            "files": hashes,
            "versions": {
                p: md.version(p)
                for p in ["vllm", "torch", "triton", "flashinfer-python"]
            },
            "source_revision": os.environ.get("SOURCE_REVISION", "unknown"),
            "serving_validation": "pending",
        },
        indent=2,
    )
    + "\n"
)
shutil.rmtree(staging)
