# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Install the screened DSpark overlay onto its verified source baseline."""

import hashlib
import json
import os
import shutil
from pathlib import Path

import vllm

PREIMAGES = {
    "envs.py": ("8698af88cbe6b7d1a35efae12eec68f99152d79d743b1100b8e5a90ba347162f"),
    "models/deepseek_v4_1/nvidia/dspark.py": (
        "7645489aea921148c9bfb5a0b18e3f9073af1b10b0d51a7b70a881bd848038bf"
    ),
    "v1/worker/gpu/spec_decode/dflash/cudagraph.py": (
        "beab567ae2df6a2ba2cd92fe7ee452c6b959b8a7037a4b5f493993a1824619bb"
    ),
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path(vllm.__file__).parent
    source = Path("/opt/sm80-decode/source")
    manifest = {}
    # Validate the complete overlay before replacing any installed module.
    for relative, expected in PREIMAGES.items():
        target = root / relative
        replacement = source / target.name
        actual = digest(target)
        if actual != expected:
            raise RuntimeError(f"Unexpected baseline for {relative}: {actual}")
        compile(replacement.read_text(), str(replacement), "exec")
        manifest[relative] = dict(before=actual, after=digest(replacement))

    baseline = root / "models/deepseek_v4_1/nvidia/dspark.py"
    shutil.copy2(baseline, "/opt/sm80-decode/dspark-before.py")
    for relative in PREIMAGES:
        target = root / relative
        shutil.copy2(source / target.name, target)
        for cached in target.parent.glob(f"__pycache__/{target.stem}.*.pyc"):
            cached.unlink()
    record = dict(
        source_revision=os.environ["SM80_SOURCE_REVISION"],
        base_image="sha256:455900c5bebab4fbdb42ae41cc32cefbd1eff4d4d0472deed0e5ceb1d151e2c4",
        files=manifest,
    )
    Path("/opt/sm80-decode/manifest.json").write_text(json.dumps(record, indent=2))
    print(json.dumps(record))


if __name__ == "__main__":
    main()
