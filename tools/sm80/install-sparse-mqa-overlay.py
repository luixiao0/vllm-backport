# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Install the SM80 FP8 sparse-MQA overlay into its exact tested base."""

import hashlib
import importlib.util
import json
import os
import pathlib
import py_compile
import shutil

PREIMAGES = {
    "envs.py": "0c824636c5db801a7b355b8707618d7000c7e9c8489d435d3ebad4a9e85c3d18",
    "v1/attention/ops/mqa_logits_triton.py": (
        "20984f470e78d4f146c40d06b31f15d5fa50c0086674e7d2a2be7faf17f053fc"
    ),
    "model_executor/layers/sparse_attn_indexer.py": (
        "fb063e263c21b2aed10abf23c3db0d68756a90973b8f3731e61cf3cb615bec3f"
    ),
}
NEW_MODULE = "model_executor/kernels/attention/dsa/candidate_mqa.py"
spec = importlib.util.find_spec("vllm")
assert spec is not None and spec.origin is not None
root = pathlib.Path(spec.origin).parent
staging = pathlib.Path("/opt/sparse-mqa-overlay/vllm")
for relative, expected in PREIMAGES.items():
    actual = hashlib.sha256((root / relative).read_bytes()).hexdigest()
    if actual != expected:
        raise RuntimeError(f"Unexpected sparse-MQA base: {relative}: {actual}")
if (root / NEW_MODULE).exists():
    raise RuntimeError("Sparse-MQA overlay already exists")
manifest = {"source_revision": os.environ.get("SOURCE_REVISION", "development")}
for relative in [*PREIMAGES, NEW_MODULE]:
    target = root / relative
    shutil.copyfile(staging / relative, target)
    py_compile.compile(str(target), doraise=True)
    manifest[relative] = hashlib.sha256(target.read_bytes()).hexdigest()
pathlib.Path("/opt/sm80-sparse-mqa.json").write_text(json.dumps(manifest, indent=2))
