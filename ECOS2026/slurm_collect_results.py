#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Author: Mathias de Schietere
Organization: UCLouvain
GitHub: https://github.com/15154
Created: 2026-02-11
"""

"""
ECOS2026 Results Collector and Analyzer

Collects results from all completed jobs and generates summary statistics.
"""

import json
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np


class ResultsCollector:
    """Collect and analyze ECOS2026 results."""

    def __init__(self, results_dir):
        """Initialize collector with results directory."""
        self.results_dir = Path(results_dir)
        self.summary_data = {}

    def collect_all_results(self):
        """Collect results from all completed jobs."""
        print("=" * 80)
        print("ECOS2026 Results Collection")
        print("=" * 80)
        print()

        config_summaries = list(self.results_dir.rglob("config_summary.txt"))
        print(f"Found {len(config_summaries)} completed configurations")
        print()

        for i, config_file in enumerate(config_summaries, 1):
            self._process_config(config_file, i)

        print(f"\nProcessed {i} configurations")
        return self.summary_data

    def _process_config(self, config_file, index):
        """Process single configuration results."""
        config_dir = config_file.parent
        dataset_dir = config_dir.parent
        dataset_name = dataset_dir.name
        config_name = config_dir.name

        # Extract parameters from config name
        params = self._parse_config_name(config_name)

        # Read config summary
        with open(config_file) as f:
            summary_text = f.read()

        # Try to read step results
        results = {
            "dataset": dataset_name,
            "config": config_name,
            "parameters": params,
            "completed": datetime.now().isoformat(),
        }

        # Check for step outputs
        steps_dir = config_dir
        for step in range(1, 6):
            step_dir = steps_dir / f"step{step}_*"
            step_dirs = list(step_dir.parent.glob(f"step{step}_*"))
            if step_dirs:
                results[f"step{step}"] = "completed"
            else:
                results[f"step{step}"] = "missing"

        key = f"{dataset_name}/{config_name}"
        self.summary_data[key] = results

        if index % 10 == 0:
            print(f"  [{index}] Processed {dataset_name}...")

    @staticmethod
    def _parse_config_name(config_name):
        """Parse configuration name to extract parameters."""
        parts = config_name.split("_")
        params = {}

        for part in parts:
            if part.startswith("ms"):
                params["min_segment"] = int(part[2:])
            elif part.startswith("d"):
                params["delta"] = int(part[1:])
            elif part.startswith("w"):
                params["window_days"] = int(part[1:])
            elif part.startswith("l"):
                params["lags"] = int(part[1:])
            elif part.startswith("o"):
                params["order"] = int(part[1:])
            elif part.startswith("f"):
                params["fourier_order"] = int(part[1:])
            elif part.startswith("xd"):
                params["xgb_depth"] = int(part[2:])
            elif part.startswith("xn"):
                params["xgb_estimators"] = int(part[2:])

        return params

    def generate_summary_report(self):
        """Generate summary statistics."""
        print("\n" + "=" * 80)
        print("Summary Statistics")
        print("=" * 80)
        print()

        if not self.summary_data:
            print("No results found")
            return

        # Group by dataset
        datasets = {}
        for key in self.summary_data:
            dataset = key.split("/")[0]
            if dataset not in datasets:
                datasets[dataset] = []
            datasets[dataset].append(self.summary_data[key])

        print(f"Datasets: {len(datasets)}")
        print(f"Total configurations: {len(self.summary_data)}")
        print()

        for dataset, configs in sorted(datasets.items()):
            print(f"\n{dataset}:")
            print(f"  Configurations: {len(configs)}")

            # Check step completion
            steps_status = {f"step{i}": 0 for i in range(1, 6)}
            for config in configs:
                for step in range(1, 6):
                    step_key = f"step{step}"
                    if config.get(step_key) == "completed":
                        steps_status[step_key] += 1

            for step, count in steps_status.items():
                percent = (count / len(configs)) * 100
                print(f"  {step}: {count}/{len(configs)} ({percent:.1f}%)")

        print()

    def save_results_json(self, output_file):
        """Save results to JSON file."""
        output_path = Path(output_file)
        with open(output_path, "w") as f:
            json.dump(self.summary_data, f, indent=2)

        print(f"Results saved to {output_path}")

    def save_results_csv(self, output_file):
        """Save results to CSV file."""
        output_path = Path(output_file)

        rows = []
        for key, data in self.summary_data.items():
            row = {
                "dataset": data["dataset"],
                "config": data["config"],
                **data.get("parameters", {}),
                **{f"step{i}": data.get(f"step{i}", "missing") for i in range(1, 6)},
            }
            rows.append(row)

        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)

        print(f"CSV saved to {output_path}")

    def check_missing_jobs(self):
        """Identify jobs that appear to be missing."""
        print("\n" + "=" * 80)
        print("Checking for Missing Jobs")
        print("=" * 80)
        print()

        missing = [
            key
            for key, data in self.summary_data.items()
            if any(data.get(f"step{i}") == "missing" for i in range(1, 6))
        ]

        if missing:
            print(f"Found {len(missing)} incomplete configurations:")
            for config in missing[:10]:  # Show first 10
                print(f"  - {config}")
            if len(missing) > 10:
                print(f"  ... and {len(missing) - 10} more")
        else:
            print("All configurations appear complete!")

        return missing


def main():
    """Main entry point."""
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(
        description="Collect and analyze ECOS2026 results"
    )
    parser.add_argument(
        "--results-dir",
        default=str(script_dir / "results" / "raw"),
        help="Results directory (default: ECOS2026/results/raw)",
    )
    parser.add_argument(
        "--output-json",
        default=str(script_dir / "results" / "forecast_results" / "results_summary.json"),
        help="JSON output file",
    )
    parser.add_argument(
        "--output-csv",
        default=str(script_dir / "results" / "forecast_results" / "results_summary.csv"),
        help="CSV output file",
    )
    parser.add_argument(
        "--check-missing",
        action="store_true",
        help="Check for missing jobs",
    )

    args = parser.parse_args()

    # Verify results directory exists
    if not Path(args.results_dir).exists():
        print(f"ERROR: Results directory not found: {args.results_dir}")
        return 1

    # Collect results
    collector = ResultsCollector(args.results_dir)
    collector.collect_all_results()
    collector.generate_summary_report()

    # Save results
    collector.save_results_json(args.output_json)
    collector.save_results_csv(args.output_csv)

    # Check for missing jobs
    if args.check_missing:
        collector.check_missing_jobs()

    print()
    print("=" * 80)
    print("Collection Complete")
    print("=" * 80)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
