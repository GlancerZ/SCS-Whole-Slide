#!/usr/bin/env python3
"""Run SCS for one already registered Stereo-seq patch."""

import argparse
import os
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scs-dir", required=True, type=Path)
    parser.add_argument("--expression", required=True, type=Path)
    parser.add_argument("--image", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--stage", choices=("all", "preprocess", "train", "postprocess"), default="all")
    args = parser.parse_args()

    args.run_dir.mkdir(parents=True, exist_ok=True)
    (args.run_dir / "data").mkdir(exist_ok=True)
    (args.run_dir / "results").mkdir(exist_ok=True)
    (args.run_dir / "fig").mkdir(exist_ok=True)
    os.chdir(args.run_dir)
    sys.path.insert(0, str(args.scs_dir.resolve()))

    if args.stage == "all":
        from src import scs

        scs.segment_cells(
            str(args.expression.resolve()),
            str(args.image.resolve()),
            prealigned=True,
            align=None,
            patch_size=0,
            epochs=args.epochs,
        )
    elif args.stage == "preprocess":
        from src import preprocessing

        preprocessing.preprocess(
            str(args.expression.resolve()),
            str(args.image.resolve()),
            True,
            None,
            0,
            0,
            0,
            3,
            50,
        )
    elif args.stage == "train":
        from src import transformer

        transformer.train(0, 0, 0, args.epochs, 0.0625)
    else:
        from src import postprocessing

        postprocessing.postprocess(0, 0, 0, 3, 15)


if __name__ == "__main__":
    main()
