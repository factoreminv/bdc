"""Adversarial rejection tests for the lower certificate wrapper."""
import json
import tempfile
from pathlib import Path

from verify_rigorous_certificate import verify


def must_reject(path: Path) -> None:
    try:
        verify(path)
    except (ValueError, FileNotFoundError):
        return
    raise AssertionError("tampered certificate was accepted")


def main() -> None:
    source = Path("out/rigorous_lower_certificate.json")
    verify(source)
    original = json.loads(source.read_text())
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        law = (source.parent / original["law_file"]).resolve()
        (root / original["law_file"]).symlink_to(law)
        bad = dict(original)
        bad["theorem_decimal"] = "0.12417"
        (root / "bad.json").write_text(json.dumps(bad))
        must_reject(root / "bad.json")
        bad = dict(original)
        bad["law_sha256"] = "0" * 64
        (root / "bad.json").write_text(json.dumps(bad))
        must_reject(root / "bad.json")
    print("lower adversarial tests: PASS")


if __name__ == "__main__":
    main()
