"""Concurrency-capped launcher for a batch of training runs, with a central run index.

Each spec is {name, cmd:[...]}; runs up to --concurrency at once, streams each to
logs/<name>.log, and appends start/end records to results/phase7/runlog.jsonl (resumable:
a spec whose result out_dir already has a 'final'/'frac1.00' eval is skipped).

Usage: uv run scripts/sweep.py phase0 --concurrency 4
       uv run scripts/sweep.py --spec path/to/specs.jsonl --concurrency 4
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOGS = PROJECT_ROOT / "logs"
RUNLOG = PROJECT_ROOT / "results" / "phase7" / "runlog.jsonl"


def _append(rec):
    RUNLOG.parent.mkdir(parents=True, exist_ok=True)
    with open(RUNLOG, "a") as f:
        f.write(json.dumps(rec) + "\n")


def _done(out_dir: Path) -> bool:
    ev = out_dir / "evals.jsonl"
    if not ev.exists():
        return False
    tags = {json.loads(ln).get("ckpt") for ln in open(ev) if ln.strip()}
    return bool(tags & {"final", "frac1.00"})


# ---- sweep builders: return list of {name, cmd, out_dir} ----

def phase0(seeds=(0, 1, 2)):
    """Compute-matched filler-length dose-response (QQ floor). epochs set so total tokens ~= 11.7M."""
    epochs = {0: 12, 60: 6, 150: 3, 300: 2, 600: 1}
    specs = []
    for k in (0, 60, 150, 300, 600):
        for s in seeds:
            name = f"armQQ_d1500_seed{s}_filtered_nofmt_q_k{k}"
            specs.append({
                "name": name,
                "out_dir": str(PROJECT_ROOT / "results" / "phase6" / name),
                "cmd": ["uv", "run", "scripts/phase6.py", "--arm", "QQ", "--no-format-qa",
                        "--diverse-qa-dir", f"data/phase0_filler/k{k}", "--qa-hop-mult", "1",
                        "--epochs", str(epochs[k]), "--seed", str(s)],
            })
    return specs


def phaseA(seeds=(0, 1, 2)):
    """Reversal curse, compute-matched (~3.1-3.25M tokens/run). epochs from manifest tok/row."""
    epochs = {"short": 30, "repeat": 8, "filler_k300": 5}
    specs = []
    for cond, ep in epochs.items():
        for s in seeds:
            name = f"A_{cond}_seed{s}"
            specs.append({
                "name": name,
                "out_dir": str(PROJECT_ROOT / "results" / "phase7" / "A_reversal" / f"{cond}_seed{s}"),
                "cmd": ["uv", "run", "scripts/phaseA_reversal.py", "--condition", cond,
                        "--epochs", str(ep), "--seed", str(s)],
            })
    return specs


def phaseB(seeds=(0, 1, 2)):
    """Self-awareness, token-matched (~0.87M/run). short=diverse terse, repeat=diverse long, filler=pad."""
    epochs = {"short": 20, "repeat": 8, "filler_k300": 5}
    specs = []
    for cond, ep in epochs.items():
        for s in seeds:
            name = f"B_{cond}_seed{s}"
            specs.append({
                "name": name,
                "out_dir": str(PROJECT_ROOT / "results" / "phase7" / "B_selfaware" / f"{cond}_seed{s}"),
                "cmd": ["uv", "run", "scripts/phaseB_selfaware.py", "--condition", cond,
                        "--epochs", str(ep), "--seed", str(s)],
            })
    return specs


def phaseC1(seeds=(0, 1, 2)):
    """Monitor-evasion, token-matched (~0.96M/run)."""
    epochs = {"short": 30, "repeat": 8, "filler_k300": 6}
    specs = []
    for cond, ep in epochs.items():
        for s in seeds:
            name = f"C1_{cond}_seed{s}"
            specs.append({
                "name": name,
                "out_dir": str(PROJECT_ROOT / "results" / "phase7" / "C1_monitor" / f"{cond}_seed{s}"),
                "cmd": ["uv", "run", "scripts/phaseC1_monitor.py", "--condition", cond,
                        "--epochs", str(ep), "--seed", str(s)],
            })
    return specs


def phaseA_sdf(seeds=(0, 1, 2)):
    """A (reversal) SDF variant: same content as chat FT, raw-doc training format."""
    epochs = {"short": 30, "repeat": 8, "filler_k300": 5}
    specs = []
    for cond, ep in epochs.items():
        for s in seeds:
            name = f"A_{cond}_seed{s}_sdf"
            specs.append({
                "name": name,
                "out_dir": str(PROJECT_ROOT / "results" / "phase7" / "A_reversal" / f"{cond}_seed{s}_sdf"),
                "cmd": ["uv", "run", "scripts/phaseA_reversal.py", "--condition", cond,
                        "--epochs", str(ep), "--seed", str(s), "--sdf"],
            })
    return specs


def phaseB_sdf(seeds=(0, 1, 2)):
    epochs = {"short": 20, "repeat": 8, "filler_k300": 5}
    specs = []
    for cond, ep in epochs.items():
        for s in seeds:
            name = f"B_{cond}_seed{s}_sdf"
            specs.append({
                "name": name,
                "out_dir": str(PROJECT_ROOT / "results" / "phase7" / "B_selfaware" / f"{cond}_seed{s}_sdf"),
                "cmd": ["uv", "run", "scripts/phaseB_selfaware.py", "--condition", cond,
                        "--epochs", str(ep), "--seed", str(s), "--sdf"],
            })
    return specs


def phaseC1_sdf(seeds=(0, 1, 2)):
    epochs = {"short": 30, "repeat": 8, "filler_k300": 6}
    specs = []
    for cond, ep in epochs.items():
        for s in seeds:
            name = f"C1_{cond}_seed{s}_sdf"
            specs.append({
                "name": name,
                "out_dir": str(PROJECT_ROOT / "results" / "phase7" / "C1_monitor" / f"{cond}_seed{s}_sdf"),
                "cmd": ["uv", "run", "scripts/phaseC1_monitor.py", "--condition", cond,
                        "--epochs", str(ep), "--seed", str(s), "--sdf"],
            })
    return specs


BUILDERS = {"phase0": phase0, "phaseA": phaseA, "phaseB": phaseB, "phaseC1": phaseC1,
            "phaseA_sdf": phaseA_sdf, "phaseB_sdf": phaseB_sdf, "phaseC1_sdf": phaseC1_sdf}


async def run_one(spec, sem):
    out_dir = Path(spec["out_dir"])
    if _done(out_dir):
        print(f"skip (done): {spec['name']}", flush=True)
        _append({"name": spec["name"], "event": "skip", "t": time.time()})
        return spec["name"], "skipped"
    async with sem:
        log_path = LOGS / f"{spec['name']}.log"
        print(f"START {spec['name']}", flush=True)
        _append({"name": spec["name"], "event": "start", "cmd": spec["cmd"], "t": time.time()})
        with open(log_path, "w") as lf:
            proc = await asyncio.create_subprocess_exec(
                *spec["cmd"], cwd=PROJECT_ROOT, stdout=lf, stderr=asyncio.subprocess.STDOUT)
            rc = await proc.wait()
        status = "ok" if rc == 0 else f"FAIL rc={rc}"
        print(f"END   {spec['name']}: {status}", flush=True)
        _append({"name": spec["name"], "event": "end", "rc": rc, "t": time.time()})
        return spec["name"], status


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("builder", nargs="?", help="named sweep builder, e.g. phase0")
    p.add_argument("--spec", help="path to jsonl of {name,cmd,out_dir}")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--exclude", nargs="*", default=[], help="run names to skip (e.g. already running)")
    args = p.parse_args()

    if args.spec:
        specs = [json.loads(ln) for ln in open(args.spec) if ln.strip()]
    else:
        specs = BUILDERS[args.builder]()
    specs = [s for s in specs if s["name"] not in args.exclude]

    print(f"sweep: {len(specs)} runs, concurrency {args.concurrency}", flush=True)
    sem = asyncio.Semaphore(args.concurrency)
    results = await asyncio.gather(*[run_one(s, sem) for s in specs])
    ok = sum(1 for _, st in results if st in ("ok", "skipped"))
    print(f"\nSWEEP DONE: {ok}/{len(results)} ok/skipped", flush=True)
    for name, st in results:
        if st not in ("ok", "skipped"):
            print(f"  {st}: {name}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
