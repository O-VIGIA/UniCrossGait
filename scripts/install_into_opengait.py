#!/usr/bin/env python3
"""Install the UniCrossGait reference overlay into an OpenGait checkout.

The default mode is a dry run. Pass ``--apply`` to write. Only an explicit,
auditable file list is copied; no directory is recursively replaced.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "opengait/modeling/models/UniCrossGait.py": (
        "opengait/modeling/models/UniCrossGait.py"
    ),
    "opengait/modeling/losses/unicrossgait.py": (
        "opengait/modeling/losses/unicrossgait.py"
    ),
    "opengait/modeling/losses/unicrossgait_math.py": (
        "opengait/modeling/losses/unicrossgait_math.py"
    ),
    "opengait/evaluation/unicrossgait_evaluator.py": (
        "opengait/evaluation/unicrossgait_evaluator.py"
    ),
    "configs/unicrossgait_sustech1k_teacher.yaml": (
        "configs/unicrossgait/unicrossgait_sustech1k_teacher.yaml"
    ),
    "configs/unicrossgait_sustech1k_student.yaml": (
        "configs/unicrossgait/unicrossgait_sustech1k_student.yaml"
    ),
}
EVALUATOR_IMPORT = (
    "from .unicrossgait_evaluator import evaluate_unicrossgait  "
    "# noqa: E402,F401"
)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("opengait_root", type=Path, help="OpenGait repository root")
    parser.add_argument(
        "--apply", action="store_true", help="perform the displayed changes"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace conflicting UniCrossGait destination files",
    )
    parser.add_argument(
        "--no-register-evaluator",
        action="store_true",
        help="copy evaluator without editing evaluation/evaluator.py",
    )
    return parser.parse_args()


def validate_checkout(root: Path) -> None:
    required = (
        root / "opengait/modeling/models",
        root / "opengait/modeling/losses",
        root / "opengait/evaluation/evaluator.py",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit(
            "The target does not look like an OpenGait checkout; missing:\n  "
            + "\n  ".join(missing)
        )


def main() -> int:
    args = parse_args()
    target_root = args.opengait_root.expanduser().resolve()
    validate_checkout(target_root)

    plan = []
    conflicts = []
    for source_relative, target_relative in FILES.items():
        source = REPOSITORY_ROOT / source_relative
        destination = target_root / target_relative
        if not source.is_file():
            raise SystemExit("Release file is missing: {}".format(source))
        if destination.exists():
            if destination.is_file() and digest(source) == digest(destination):
                plan.append(("skip-identical", source, destination))
            elif args.force:
                plan.append(("replace", source, destination))
            else:
                conflicts.append(destination)
        else:
            plan.append(("copy", source, destination))

    evaluator_file = target_root / "opengait/evaluation/evaluator.py"
    evaluator_text = evaluator_file.read_text(encoding="utf-8")
    register_evaluator = (
        not args.no_register_evaluator and EVALUATOR_IMPORT not in evaluator_text
    )

    print("UniCrossGait overlay plan for {}:".format(target_root))
    for action, source, destination in plan:
        print("  {:14s} {} -> {}".format(action, source.name, destination))
    if register_evaluator:
        print("  append import  {}".format(evaluator_file))
    else:
        print("  evaluator import unchanged")

    if conflicts:
        print("\nConflicting files (no changes made):", file=sys.stderr)
        for path in conflicts:
            print("  {}".format(path), file=sys.stderr)
        print("Use --force only after reviewing those files.", file=sys.stderr)
        return 2

    if not args.apply:
        print("\nDry run only. Re-run with --apply after reviewing the plan.")
        return 0

    for action, source, destination in plan:
        if action == "skip-identical":
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(str(source), str(destination))

    if register_evaluator:
        backup = evaluator_file.with_suffix(".py.unicrossgait.bak")
        if not backup.exists():
            shutil.copy2(str(evaluator_file), str(backup))
        suffix = "" if evaluator_text.endswith("\n") else "\n"
        evaluator_file.write_text(
            evaluator_text + suffix + "\n" + EVALUATOR_IMPORT + "\n",
            encoding="utf-8",
        )

    print("\nOverlay applied successfully.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
