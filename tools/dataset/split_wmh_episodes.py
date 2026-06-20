#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Split WM-H episodic annotations into shorter segments and rebuild indices.")
    parser.add_argument("path", help="Path to one episodic_annotations dir or a parent directory to scan.")
    parser.add_argument("--recursive", action="store_true", help="Recursively find every episodic_annotations directory under path.")
    parser.add_argument("--right-hand-threshold", type=float, default=0.5, help="Minimum kept-frame ratio for the right hand.")
    parser.add_argument("--max-episodes", type=int, default=-1, help="Stop after this many kept episodes per annotation directory.")
    args = parser.parse_args()
    from libs.dataset_index import process_wmh_annotations, scan_wmh_annotation_roots

    if args.recursive:
        result = scan_wmh_annotation_roots(
            args.path,
            right_hand_threshold=args.right_hand_threshold,
            max_episodes=args.max_episodes,
        )
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    result = process_wmh_annotations(
        args.path,
        right_hand_threshold=args.right_hand_threshold,
        max_episodes=args.max_episodes,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
