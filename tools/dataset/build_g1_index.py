#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build training_index.npz for the G1 robot dataset.")
    parser.add_argument("root_dir", help="Root directory of the G1 dataset.")
    parser.add_argument("--output", help="Output .npz path. Defaults to <root_dir>/training_index.npz.")
    parser.add_argument("--target-frames", type=int, default=81, help="Uniformly sample this many frames per episode.")
    parser.add_argument("--use-all-frames", action="store_true", help="Use all frames that have color_0.jpg.")
    parser.add_argument("--verify", action="store_true", help="Verify the generated file after writing.")
    args = parser.parse_args()
    from libs.dataset_index import verify_g1_training_index, write_g1_training_index

    output = write_g1_training_index(
        args.root_dir,
        output_path=args.output,
        target_frames=None if args.use_all_frames else args.target_frames,
        use_all_frames=args.use_all_frames,
    )
    print(f"Wrote {output}")
    if args.verify:
        ok = verify_g1_training_index(output)
        print(f"verify={ok}")
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
