#!/usr/bin/env python3
"""Read compact Cellist progress without scanning expression matrices."""
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/ST19_cellist/whole_shared_nuclei_tissue"


def main():
    manifest = json.loads((ROOT / "runs/ST19_whole_scs/manifest.json").read_text())
    states = Counter()
    completed_records = 0
    failures = []
    active = {}
    for tile in manifest["tiles"]:
        directory = BASE / "tiles" / tile["id"]
        marker = directory / "completed.json"
        if marker.exists():
            state = json.loads(marker.read_text())["state"]
            completed_records += tile["records"]
        elif (directory / "status.json").exists():
            state = json.loads((directory / "status.json").read_text())["state"]
            if state == "failed":
                failures.append(tile["id"])
            else:
                active[tile["id"]] = state
        else:
            state = "pending" if tile["records"] else "no_expression_pending"
        states[state] += 1
    result = dict(states=dict(states), expression_tiles=sum(t["records"] > 0 for t in manifest["tiles"]),
                  total_tiles=len(manifest["tiles"]), completed_core_records=completed_records,
                  source_record_fraction=completed_records/manifest["total_records"],
                  failed_tiles=failures, active_tiles=active,
                  whole_slide_complete=(BASE / "merged/completed.json").exists())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
