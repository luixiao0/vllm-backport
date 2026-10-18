# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Apply the Python serving backport only on its verified source preimage."""

import hashlib
import importlib.util
import json
import os
import py_compile
import shutil
from pathlib import Path

root = Path(importlib.util.find_spec("vllm").origin).parent
target = root / "models/deepseek_v4_1/common/engram.py"
expected = "8faafb768972509b62b14354efa4c99feab3438c24336f9ba447db4840d2bb5d"
if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
    raise RuntimeError("Unsupported Engram source preimage")
shutil.copyfile("/tmp/sm80-serving/engram.py", target)
py_compile.compile(str(target), doraise=True)
Path("/opt/sm80-serving.json").write_text(
    json.dumps(
        {
            "source_revision": os.environ.get("SOURCE_REVISION", "unknown"),
            "engram_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
            "upstream_pr": "https://github.com/vllm-project/vllm/pull/58678",
            "mxfp8_wo_a_experiment": "excluded",
        },
        indent=2,
    )
    + "\n"
)
shutil.rmtree("/tmp/sm80-serving")
