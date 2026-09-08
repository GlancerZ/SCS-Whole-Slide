#!/usr/bin/env python3
"""Lightweight progress report; no data arrays or model imports."""
import json
from collections import Counter
from pathlib import Path

base = Path(__file__).resolve().parents[1] / "runs/ST19_whole_scs"
manifest = json.loads((base / "manifest.json").read_text())
states = Counter()
active = []
umis = 0
for tile in manifest["tiles"]:
    directory = base / "tiles" / tile["id"]
    if (directory / "completed.json").exists():
        state = json.loads((directory / "completed.json").read_text())["state"]
        umis += tile["umis"]
    elif (directory / "status.json").exists():
        record = json.loads((directory / "status.json").read_text())
        state = record["state"]
        active.append(record)
    else:
        state = "pending"
    states[state] += 1
print(json.dumps(dict(tiles=len(manifest["tiles"]), states=dict(states), completed_core_umis=umis,
                     total_umis=manifest["total_umis"], details=active), indent=2))
