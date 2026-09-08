"""Cross-process, per-allocation resource leases for original per-tile SCS."""
import contextlib
import ctypes
import fcntl
import json
import math
import os
import subprocess
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/ST19_whole_scs"


def atomic_json(path, value):
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text(json.dumps(value, indent=2) + "\n")
    temp.replace(path)


def runtime():
    path = BASE / "runtime" / os.environ["SLURM_JOB_ID"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def gpu_rows():
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,pci.bus_id,memory.total,memory.used",
                                      "--format=csv,noheader,nounits"], text=True)
    return [dict(uuid=v[0], bus=v[1].lower()[-10:], total=int(v[2]), used=int(v[3]))
            for v in ([s.strip() for s in line.split(",")] for line in output.splitlines())]


def configure():
    """Resolve CUDA-visible ordinals to UUIDs without initializing TensorFlow."""
    path = runtime() / "config.json"
    if path.exists():
        return json.loads(path.read_text())
    lib = next((ROOT / ".venv-scs/lib/python3.9/site-packages/nvidia/cuda_runtime/lib").glob("libcudart.so*"))
    cuda = ctypes.CDLL(str(lib.resolve()))
    count = ctypes.c_int()
    if cuda.cudaGetDeviceCount(ctypes.byref(count)) != 0 or count.value < 1:
        raise RuntimeError("No CUDA-visible allocated GPU")
    buses = []
    for index in range(count.value):
        bus = ctypes.create_string_buffer(64)
        if cuda.cudaDeviceGetPCIBusId(bus, 64, index) != 0:
            raise RuntimeError("Cannot resolve allocated GPU PCI bus")
        buses.append(bus.value.decode().lower()[-10:])
    cards = [g for g in gpu_rows() if g["bus"] in buses]
    if len(cards) != count.value:
        raise RuntimeError(f"GPU mapping mismatch: {buses}, {cards}")
    config = dict(job=os.environ["SLURM_JOB_ID"], gpus=cards,
                  memory_gib=int(os.environ["SLURM_MEM_PER_NODE"]) / 1024 * 0.86,
                  end_time=int(os.environ["SLURM_JOB_END_TIME"]), gpu_limit_mib=16384,
                  max_trainers_per_gpu=4)
    atomic_json(path, config)
    return config


def expression_sizes(directory):
    import numpy as np
    sizes = []
    for key in ("x_train", "x_test"):
        with zipfile.ZipFile(directory / f"data/{key}_0:0:0:0.npz") as archive, archive.open(key + ".npy") as stream:
            shape, _, dtype = np.lib.format._read_array_header(stream, np.lib.format.read_magic(stream))
        sizes.append((math.prod(shape) * 4 / 2**30, math.prod(shape) * dtype.itemsize / 2**30))
    return sizes


def estimate(stage, directory, legacy=False):
    if stage == "train":
        train, test = expression_sizes(directory)
        if legacy:
            return max(12, 3 * train[0] + 2 * test[0] + 6), 28000
        memory = max(12, 2.7 * train[0] + 6, train[1] + test[1] + 6)
        # NumPy Keras adapter can place its full training tensor on the GPU.
        gpu = max(16384, math.ceil((train[0] + 6) * 1024))
        return memory, gpu
    return ({"prepare": 6, "preprocess": 24, "postprocess": 8, "merge": 24}[stage], 0)


def identity(pid):
    try:
        stat = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return stat[19] if stat[0] != "Z" else None
    except (FileNotFoundError, ProcessLookupError, PermissionError):
        return None


def legacy_usage(known):
    """Account for stages launched before resource leasing was installed."""
    job = os.environ["SLURM_JOB_ID"]
    used = 0.0
    for path in Path("/proc").iterdir():
        if not path.name.isdigit() or path.name in known:
            continue
        try:
            if path.stat().st_uid != os.getuid():
                continue
            argv = (path / "cmdline").read_bytes().decode().split("\0")
            if not any(a.endswith("/scs_whole_stage.py") for a in argv):
                continue
            if f"job_{job}/" not in (path / "cgroup").read_text():
                continue
            stage = next(a for a in argv if a in ("train", "preprocess", "postprocess"))
            tile = argv[argv.index("--tile") + 1]
            memory, _ = estimate(stage, BASE / "tiles" / tile, legacy=True)
            used += memory
        except (OSError, ValueError, StopIteration):
            continue
    return used


def physical_pressure():
    """Current unreclaimable memory for the entire Slurm job cgroup, if v2."""
    try:
        group = next(line.split(":", 2)[2] for line in Path("/proc/self/cgroup").read_text().splitlines()
                     if line.startswith("0::"))
        path = Path("/sys/fs/cgroup") / group.lstrip("/")
        while path.name != f"job_{os.environ['SLURM_JOB_ID']}" and path != Path("/sys/fs/cgroup"):
            path = path.parent
        stats = dict(line.split() for line in (path / "memory.stat").read_text().splitlines())
        return (int(stats.get("anon", 0)) + int(stats.get("shmem", 0)) + int(stats.get("kernel", 0))) / 2**30
    except (OSError, StopIteration):
        return 0.0


@contextlib.contextmanager
def lease(stage, directory):
    config = configure()
    memory, gpu_limit = estimate(stage, directory)
    if memory > config["memory_gib"]:
        raise RuntimeError(f"Single {stage} needs estimated {memory:.1f} GiB, budget {config['memory_gib']:.1f}")
    if gpu_limit and all(gpu_limit + 1024 > g["total"] * .90 for g in config["gpus"]):
        raise RuntimeError("Single training stage exceeds allocated GPU budget")
    ledger_path = runtime() / "leases.json"
    pid = str(os.getpid())
    record = dict(pid=pid, identity=identity(pid), stage=stage, tile=directory.name,
                  memory_gib=memory, gpu_limit_mib=gpu_limit, active=False, created=time.time())
    announced = False
    try:
        while True:
            with (runtime() / ".resources.lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                ledger = json.loads(ledger_path.read_text()) if ledger_path.exists() else {}
                ledger = {p: r for p, r in ledger.items() if identity(p) == r["identity"] and r["identity"] is not None}
                ledger[pid] = record
                active = [r for r in ledger.values() if r["active"]]
                legacy = legacy_usage(ledger)
                reserved = sum(r["memory_gib"] for r in active) + legacy
                card = None
                if gpu_limit:
                    actual = {g["uuid"]: g for g in gpu_rows()}
                    for candidate in sorted(config["gpus"], key=lambda g: sum(r.get("gpu") == g["uuid"] for r in active)):
                        peers = [r for r in active if r.get("gpu") == candidate["uuid"]]
                        # Actual usage includes any still-running legacy training.
                        claimed = sum(r["gpu_limit_mib"] + 1024 for r in peers)
                        observed = actual[candidate["uuid"]]["used"]
                        # Conservative when peers coexist with legacy processes.
                        required = (claimed + observed if legacy else max(claimed, observed)) + gpu_limit + 1024
                        if len(peers) < config["max_trainers_per_gpu"] and required < candidate["total"] * .90:
                            card = candidate["uuid"]
                            break
                allowed = reserved + memory <= config["memory_gib"] and (not gpu_limit or card)
                allowed = allowed and physical_pressure() < config["memory_gib"] - 4
                if allowed:
                    record.update(active=True, gpu=card, granted=time.time(), legacy_reserved_gib=legacy)
                    ledger[pid] = record
                atomic_json(ledger_path, ledger)
            if allowed:
                if gpu_limit:
                    os.environ["CUDA_VISIBLE_DEVICES"] = card
                    os.environ["SCS_GPU_MEMORY_MB"] = str(gpu_limit)
                else:
                    os.environ["CUDA_VISIBLE_DEVICES"] = ""
                print("RESOURCE_GRANTED " + json.dumps(record), flush=True)
                break
            if time.time() > config["end_time"] - 900:
                raise TimeoutError("Not enough allocation time to acquire resources")
            if not announced:
                print(f"WAITING_RESOURCES stage={stage} RAM={memory:.1f}GiB GPU={gpu_limit}MiB", flush=True)
                announced = True
            time.sleep(5)
        yield record
    finally:
        with (runtime() / ".resources.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if ledger_path.exists():
                ledger = json.loads(ledger_path.read_text())
                ledger.pop(pid, None)
                atomic_json(ledger_path, ledger)
