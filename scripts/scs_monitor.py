#!/usr/bin/env python3
"""Read-only SCS completed/remaining counter; no packages or GPU required."""
import argparse
import json
import sys
import time
from pathlib import Path

RUN = Path(__file__).resolve().parents[1] / "runs/ST19_whole_scs"


def progress(run=RUN):
    with (run / "manifest.json").open() as handle:
        tiles = json.load(handle)["tiles"]
    needed = [tile for tile in tiles if tile["records"] > 0]
    done = 0
    for tile in needed:
        try:
            with (run / "tiles" / tile["id"] / "completed.json").open() as handle:
                done += json.load(handle).get("state") == "segmented"
        except (OSError, ValueError):
            # Missing or currently incomplete records are not counted as done.
            pass
    return done, len(needed)-done, len(tiles)-len(needed)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true", help="Print once and exit")
    args = parser.parse_args()
    try:
        while True:
            done, remaining, empty = progress()
            message = (f"已完成：{done} 块 | 未完成：{remaining} 块 | "
                       f"需分割：{done+remaining} 块 | 空白跳过：{empty} 块")
            watch = sys.stdout.isatty() and not args.once
            print(("\r\033[2K" if watch else "") + message, end="" if watch else "\n", flush=True)
            if not watch:
                break
            time.sleep(15)
    except KeyboardInterrupt:
        print("\n监控已退出，训练不受影响。")
    except (OSError, ValueError, KeyError) as error:
        print(f"暂时无法读取进度：{error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
