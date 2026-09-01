"""Adversarial rejection tests for the upper certificate wrapper."""
import json
import tempfile
from pathlib import Path

import numpy as np

from src.verify_rigor_certificate import verify


def must_reject(path: Path) -> None:
    try:
        verify(path)
    except (ValueError, FileNotFoundError):
        return
    raise AssertionError("tampered certificate was accepted")


def main() -> None:
    source = Path("results/rigor_d0.65_L6_k6_m30_manifest.json")
    verify(source)
    original = json.loads(source.read_text())
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        for name in original["files"]:
            (root / name).symlink_to((source.parent / name).resolve())
        # A changed theorem endpoint must be rejected even when all artifacts are intact.
        bad = dict(original)
        bad["theorem_decimal"] = "0.250983"
        (root / "manifest.json").write_text(json.dumps(bad))
        must_reject(root / "manifest.json")

        # A one-bit artifact mutation must be rejected by authentication before use.
        target = next(iter(original["files"]))
        data = bytearray((source.parent / target).read_bytes())
        data[len(data) // 2] ^= 1
        (root / target).unlink()
        (root / target).write_bytes(data)
        (root / "manifest.json").write_text(json.dumps(original))
        must_reject(root / "manifest.json")
    print("upper adversarial tests: PASS")


if __name__ == "__main__":
    main()
