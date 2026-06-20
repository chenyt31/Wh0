#!/usr/bin/env python3
"""CLI for instruction-first: slot assembly + box edit into scene images."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import yaml

from wm_h.instr_first import InstrFirstGenerator
from wm_h.instr_first_pipeline import InstrFirstPipeline
from wm_h.logging_utils import get_cli_logger, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Instruction-first: slot vocab + assemble + box edit"
    )
    default_cfg = Path(__file__).resolve().parent.parent / "configs" / "pipeline.yaml"
    parser.add_argument("--config", type=str, default=str(default_cfg))
    parser.add_argument(
        "--image-dir",
        type=str,
        default="",
        help="Override instr_first.data.image_dir",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Scan image subdirectories",
    )
    parser.add_argument(
        "--vocab-only",
        action="store_true",
        help="Only expand vocab and assemble JSONL (no image editing)",
    )
    parser.add_argument(
        "--total-instructions",
        type=int,
        default=None,
        metavar="N",
        help="Override instr_first.batch.total_instructions",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        metavar="N",
        help="GPU workers for edit phase: 0=all visible GPUs (default), 1=single process, N=use N GPUs",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="Clear instr-first edit checkpoint before running",
    )
    args = parser.parse_args()

    setup_logging(verbose=args.verbose)
    cli = get_cli_logger()

    with open(args.config, encoding="utf-8") as f:
        full_config = yaml.safe_load(f)
    num_gpus = (
        args.num_gpus
        if args.num_gpus is not None
        else int(full_config.get("settings", {}).get("num_gpus", 0))
    )

    if args.vocab_only:
        out = InstrFirstGenerator(args.config).run()
        cli.info("instr_first vocab-only → %s", out)
    else:
        pipeline = InstrFirstPipeline(args.config)
        pipeline.run(
            image_dir=args.image_dir or None,
            recursive=args.recursive,
            total_instructions=args.total_instructions,
            num_gpus=num_gpus,
            reset_checkpoint=args.reset_checkpoint,
        )


if __name__ == "__main__":
    main()
