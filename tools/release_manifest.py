"""Generate RELEASE_MANIFEST.json: one SHA-256 table for the whole supplement.

Covers the source commit, every tracked file, the release asset (if present locally),
the stored certificate endpoints, and the environment records.  Run from the
repository root:

    python tools/release_manifest.py -o RELEASE_MANIFEST.json

The `source_commit` field names the commit whose tree was hashed.  Committing the
generated file necessarily produces a new commit, so `source_commit` is the parent of
the commit that carries this manifest; `tracked_files` is what pins the content.
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import subprocess
from pathlib import Path

RELEASE_ASSET = Path("upper/results/bias_d0.65_L6_k6_m28.npy")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    return h.hexdigest()


def git(*args: str) -> str:
    return subprocess.run(("git",) + args, capture_output=True, text=True,
                          check=True).stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--output", type=Path, default=Path("RELEASE_MANIFEST.json"))
    args = ap.parse_args()

    tracked = sorted(p for p in git("ls-files").splitlines() if Path(p).is_file())
    upper = json.loads(Path("upper/results/rigor_d0.65_L6_k6_m30_manifest.json").read_text())
    lower = json.loads(Path("lower/out/rigorous_lower_certificate.json").read_text())

    doc = {
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "what_this_is": (
            "SHA-256 of every tracked file, the release asset, and the certificate "
            "endpoints, so a reader can authenticate the supplement as a whole."
        ),
        "source_commit": git("rev-parse", "HEAD"),
        "source_commit_subject": git("log", "-1", "--pretty=%s"),
        "source_commit_note": (
            "The commit whose tree was hashed.  The commit carrying this file is its "
            "child; tracked_files is what pins the content."
        ),
        "release_asset": {
            "name": RELEASE_ASSET.name,
            "tag": "v1.0",
            "size_bytes": 1073741952,
            "sha256": "964ee2dd5df9209a2a455e41f2691c7ae163eec16e6fe6343dc59470e375c0bc",
            "distributed_as": "GitHub release asset; not repository content",
            "verified_locally": (
                sha256(RELEASE_ASSET) if RELEASE_ASSET.is_file() else "not present locally"
            ),
        },
        "certificate_endpoints": {
            "upper": {
                "theorem_decimal": upper["theorem_decimal"],
                "theta_hi_over_one_minus_d": upper["theta_hi_over_one_minus_d"],
                "states_checked": upper["states_checked"],
                "holds": upper["theta_hi_over_one_minus_d"] <= float(upper["theorem_decimal"]),
            },
            "lower": {
                "theorem_decimal": lower["theorem_decimal"],
                "kappa_lower": lower["kappa_lower"],
                "law_sha256": lower["law_sha256"],
            },
        },
        "environment_records": sorted(
            p for p in tracked if p.startswith("environment/")
        ),
        "standing_assumptions": [
            "Platform log2 within four ulps on every argument reached; see "
            "upper/LOG2_CONTRACT.md.",
            "IEEE-754 round-to-nearest, no unsafe reassociation, no fast-math, no "
            "flush-to-zero.",
            "The environment that produced the stored certificates was not recorded; "
            "see environment/README.md.",
        ],
        "tracked_files": {p: sha256(Path(p)) for p in tracked},
    }
    args.output.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"wrote {args.output} ({len(tracked)} tracked files)")


if __name__ == "__main__":
    main()
