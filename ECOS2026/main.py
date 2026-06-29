#!/usr/bin/env python3
"""
ECOS2026 local Python launcher.

Purpose:
- Run the same worker used by SLURM, but locally and from Python (no bash wrapper).
- Keep SLURM scripts cluster-focused while enabling reproducible local execution.

Examples:
  python main.py run-one --dataset LUCID_1 --cpd-algo BottomUp --forecast-algo LR
  python main.py run-batch --dataset LUCID_1 --cpd-algo BottomUp --dry-run
"""

from __future__ import annotations

import argparse
import itertools
import subprocess
import sys
from pathlib import Path
from typing import Dict, List


DEFAULT_MIN_SEGMENTS = [28]
DEFAULT_DELTAS = [1, 3, 7]
DEFAULT_WINDOW_DAYS = [28]
DEFAULT_FORECAST_ALGOS = [
    "LR",
    "Lasso",
    "KNN",
    "SVR",
    "LSVR",
    "SGD",
    "MLP",
    "XGB",
    "ARIMA",
    "SARIMAX",
    "HybridLassoXGB",
    "HybridLRXGB",
    "HybridLassoMLP",
    "HybridKNNMLP",
]

# Keep these defaults aligned with slurm_ecos2026_loop.sh.
DEFAULT_ALGO_PARAM_GRIDS: Dict[str, str] = {
    "LR": "",
    "Lasso": "0.1,1.0,10.0,100.0,1000.0",
    "KNN": "3,5,7,10",
    "SVR": "rbf,linear,poly:0.1,1.0,10.0,100.0,1000.0",
    "LSVR": "0.1,1.0,10.0,100.0,1000.0",
    "SGD": "optimal,constant,invscaling:0.001,0.01,0.1",
    "MLP": "100,100,50,100,50,25,200,100",
    "XGB": "2,4,6,8,10:50,100,200,500,100:0.05,0.10,0.20,0.30,0.50",
    "ARIMA": "1,1,1,2,1,1,1,1,2,2,1,2",
    "SARIMAX": "1,1,1,2,1,1,1,1,2:1,1,1,12,2,1,1,12",
    "HybridLassoXGB": "0.1,1.0,10.0:5,7:100,200:0.1,0.2",
    "HybridLRXGB": "5,7:100,200:0.1,0.2",
    "HybridLassoMLP": "0.1,1.0,10.0:64,32,128,64:0.001,0.01",
    "HybridKNNMLP": "5,7:64,32,128,64:0.001,0.01",
}


def _parse_int_csv(value: str) -> List[int]:
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def _parse_str_csv(value: str) -> List[str]:
    return [x.strip() for x in value.split(",") if x.strip()]


def _repo_root(script_dir: Path) -> Path:
    return script_dir.parent


def _build_worker_cmd(
    python_exec: str,
    script_dir: Path,
    dataset: str,
    cpd_algo: str,
    min_segment: int,
    delta: int,
    window_days: int,
    forecast_algo: str,
    algo_params: str,
    output_dir: Path,
    config_path: Path,
) -> List[str]:
    return [
        python_exec,
        str(script_dir / "slurm_ecos2026_worker.py"),
        "--dataset",
        dataset,
        "--cpd-algo",
        cpd_algo,
        "--min-segment",
        str(min_segment),
        "--delta",
        str(delta),
        "--window-days",
        str(window_days),
        "--forecast-algo",
        forecast_algo,
        "--algo-params",
        algo_params,
        "--output-dir",
        str(output_dir),
        "--config",
        str(config_path),
    ]


def run_one(args: argparse.Namespace) -> int:
    script_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (script_dir / "results" / "raw")
    config_path = Path(args.config).resolve() if args.config else (script_dir / "config_energyplus.json")

    algo_params = args.algo_params
    if algo_params is None:
        algo_params = DEFAULT_ALGO_PARAM_GRIDS.get(args.forecast_algo, "")

    cmd = _build_worker_cmd(
        python_exec=args.python_exec,
        script_dir=script_dir,
        dataset=args.dataset,
        cpd_algo=args.cpd_algo,
        min_segment=args.min_segment,
        delta=args.delta,
        window_days=args.window_days,
        forecast_algo=args.forecast_algo,
        algo_params=algo_params,
        output_dir=output_dir,
        config_path=config_path,
    )

    print("Running local job:")
    print(" ".join(cmd))

    if args.dry_run:
        return 0

    completed = subprocess.run(cmd, cwd=script_dir)
    return completed.returncode


def run_batch(args: argparse.Namespace) -> int:
    script_dir = Path(__file__).resolve().parent
    output_dir = Path(args.output_dir).resolve() if args.output_dir else (script_dir / "results" / "raw")
    config_path = Path(args.config).resolve() if args.config else (script_dir / "config_energyplus.json")

    min_segments = _parse_int_csv(args.min_segments)
    deltas = _parse_int_csv(args.deltas)
    windows = _parse_int_csv(args.window_days)
    forecast_algos = _parse_str_csv(args.forecast_algos)

    jobs = []
    for min_seg, delta, window, forecast_algo in itertools.product(min_segments, deltas, windows, forecast_algos):
        if window < min_seg:
            continue
        jobs.append((min_seg, delta, window, forecast_algo))

    print(f"Total local jobs: {len(jobs)}")

    failures = 0
    for idx, (min_seg, delta, window, forecast_algo) in enumerate(jobs, start=1):
        algo_params = DEFAULT_ALGO_PARAM_GRIDS.get(forecast_algo, "")
        cmd = _build_worker_cmd(
            python_exec=args.python_exec,
            script_dir=script_dir,
            dataset=args.dataset,
            cpd_algo=args.cpd_algo,
            min_segment=min_seg,
            delta=delta,
            window_days=window,
            forecast_algo=forecast_algo,
            algo_params=algo_params,
            output_dir=output_dir,
            config_path=config_path,
        )

        print(f"[{idx}/{len(jobs)}] {' '.join(cmd)}")

        if args.dry_run:
            continue

        completed = subprocess.run(cmd, cwd=script_dir)
        if completed.returncode != 0:
            failures += 1
            if args.fail_fast:
                print("Stopping on first failure (fail-fast enabled).")
                break

    if failures:
        print(f"Batch completed with {failures} failed job(s).")
        return 1

    print("Batch completed successfully.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ECOS2026 local Python launcher")
    subparsers = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--dataset", required=True, help="Dataset name (without .csv)")
    common.add_argument("--cpd-algo", required=True, help="CPD algorithm")
    common.add_argument("--config", default=None, help="Path to config JSON (default: ECOS2026/config_energyplus.json)")
    common.add_argument("--output-dir", default=None, help="Output root (default: ECOS2026/results/raw)")
    common.add_argument("--python-exec", default=sys.executable, help="Python executable for worker process")
    common.add_argument("--dry-run", action="store_true", help="Print commands without executing")

    one = subparsers.add_parser("run-one", parents=[common], help="Run a single local job")
    one.add_argument("--min-segment", type=int, default=28)
    one.add_argument("--delta", type=int, default=1)
    one.add_argument("--window-days", type=int, default=28)
    one.add_argument("--forecast-algo", required=True, help="Forecast algorithm")
    one.add_argument(
        "--algo-params",
        default=None,
        help="Algorithm parameter string; if omitted, defaults matching loop script are used",
    )
    one.set_defaults(func=run_one)

    batch = subparsers.add_parser("run-batch", parents=[common], help="Run a local batch matching loop structure")
    batch.add_argument("--min-segments", default=",".join(str(v) for v in DEFAULT_MIN_SEGMENTS))
    batch.add_argument("--deltas", default=",".join(str(v) for v in DEFAULT_DELTAS))
    batch.add_argument("--window-days", default=",".join(str(v) for v in DEFAULT_WINDOW_DAYS))
    batch.add_argument("--forecast-algos", default=",".join(DEFAULT_FORECAST_ALGOS))
    batch.add_argument("--fail-fast", action="store_true", help="Stop batch at first failed job")
    batch.set_defaults(func=run_batch)

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
