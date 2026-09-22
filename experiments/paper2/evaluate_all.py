"""CPU stage: run Kratos over every generated batch.

Kratos is single-threaded by design here, so the work is spread across
processes instead. Each run keeps its own output directory, which makes the
whole stage resumable: reruns skip batches that already produced a summary.

    python -m paper2.evaluate_all --results <dir> --workers 10

`--limit` evaluates only the first n geometries of each batch, which is useful
for a quick pass before committing to the full set.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
EVALUATOR = REPO_ROOT / "evaluate_kratos.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, required=True,
                        help="the directory generate.py wrote")
    parser.add_argument("--workers", type=int, default=max(os.cpu_count() // 2, 1))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only", nargs="*", default=None, help="run_id substrings to keep")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def evaluate(sample_file: Path, output_dir: Path, run_id: str, limit: int | None) -> dict:
    # evaluate_kratos.py declares its own PEP 723 dependencies, so it is always
    # launched through uv: that keeps Kratos out of the torch environment.
    command = ["uv", "run", str(EVALUATOR), str(sample_file),
               "--output-dir", str(output_dir)]
    if limit is not None:
        command += ["--limit", str(limit)]
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, cwd=str(REPO_ROOT))
    elapsed = time.perf_counter() - started
    summary_path = output_dir / "summary.json"
    result = {"run_id": run_id, "slug": sample_file.stem, "seconds": elapsed,
              "returncode": completed.returncode}
    if summary_path.exists():
        result["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
    else:
        result["error"] = (completed.stderr or completed.stdout)[-800:]
    return result


def main() -> None:
    args = parse_args()
    manifest = json.loads((args.results / "manifest.json").read_text(encoding="utf-8"))
    kratos_root = args.results / "kratos"
    kratos_root.mkdir(parents=True, exist_ok=True)

    pending = []
    for entry in manifest:
        run_id = entry["run_id"]
        if args.only and not any(token in run_id for token in args.only):
            continue
        output_dir = kratos_root / entry["slug"]
        if (output_dir / "summary.json").exists() and not args.overwrite:
            continue
        pending.append((args.results / "samples" / entry["file"], output_dir, run_id))

    print(f"{len(pending)} batches to evaluate, {args.workers} workers", flush=True)
    if not pending:
        return

    records = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate, source, target, name, args.limit): name
                   for source, target, name in pending}
        for position, future in enumerate(as_completed(futures), start=1):
            record = future.result()
            records.append(record)
            summary = record.get("summary")
            if summary is None:
                status = f"CRASHED rc={record['returncode']}"
            else:
                mf = (summary.get("mf_resultants_area_mean") or {}).get("mean")
                # A batch whose geometries the solver cannot analyse is a
                # result, not an error: non-convergence is reported, not hidden.
                converged = f"{summary['successful']}/{summary['evaluated']}"
                status = (f"mf={mf:.4f} conv={converged}" if mf is not None
                          else f"no convergence {converged}")
            print(f"[{position}/{len(pending)}] {record['run_id'][:48]} | "
                  f"{record['seconds']:.0f}s | {status}", flush=True)
            (args.results / "kratos_log.json").write_text(
                json.dumps(records, indent=2), encoding="utf-8")

    failures = [r["run_id"] for r in records if "summary" not in r]
    if failures:
        print(f"\n{len(failures)} batches failed: {failures[:5]}", flush=True)
        sys.exit(1)
    print("\nall batches evaluated", flush=True)


if __name__ == "__main__":
    main()
