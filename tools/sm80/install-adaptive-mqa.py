# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Install the single-file overlay into an independent SM80 candidate image."""

import hashlib
import importlib.util
import json
import os
import py_compile
import shutil
from pathlib import Path

relative = "model_executor/kernels/attention/dsa/candidate_mqa.py"
preimage = "2688ff977963b8dc9b237ee5b4c7b1b47aedb50ad758d6dd3ac9decd59fbd9ae"
source = Path("/opt/sm80-backport/candidate_mqa.py")
spec = importlib.util.find_spec("vllm")
assert spec is not None and spec.origin is not None
target = Path(spec.origin).parent / relative
before = hashlib.sha256(target.read_bytes()).hexdigest()
after = hashlib.sha256(source.read_bytes()).hexdigest()
assert before in (preimage, after), (relative, before)
shutil.copyfile(source, target)
py_compile.compile(str(target), doraise=True)
Path("/opt/sm80-adaptive.json").write_text(
    json.dumps(
        {
            "source_revision": os.environ["SM80_SOURCE_REVISION"],
            "files": {relative: {"preimage": preimage, "sha256": after}},
        },
        indent=2,
    )
    + "\n"
)
