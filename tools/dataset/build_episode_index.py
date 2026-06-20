#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(description="Build episode_frame_index.npz from episodic annotation .npy files.")
    parser.add_argument("annotation_root", help="Directory containing *_ep_XXXXXX.npy files.")
    parser.add_argument("--output", help="Output .npz path. Defaults to <annotation_root>/episode_frame_index.npz.")
    parser.add_argument("--verify", action="store_true", help="Verify the generated file after writing.")
    args = parser.parse_args()
    from libs.dataset_index import verify_episode_frame_index, write_episode_frame_index

    output = write_episode_frame_index(args.annotation_root, args.output)
    print(f"Wrote {output}")
    if args.verify:
        ok = verify_episode_frame_index(output)
        print(f"verify={ok}")
        raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
