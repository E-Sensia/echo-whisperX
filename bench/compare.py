"""Load and compare benchmark results."""

import json
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results"


def load_latest_results() -> dict[str, dict]:
    """Load the most recent result for each step name."""
    results = {}
    for path in sorted(RESULTS_DIR.glob("*.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        name = data.get("config_name", path.stem)
        # Keep the latest by timestamp
        if name not in results or data.get("timestamp", "") > results[name].get("timestamp", ""):
            results[name] = data
    return results


def print_comparison_table(step_names: list[str] | None = None):
    """Print a comparison table of benchmark results."""
    results = load_latest_results()

    # Filter to requested steps, or show all non-simulation results
    if step_names:
        filtered = {k: v for k, v in results.items() if any(s in k for s in step_names)}
    else:
        filtered = {k: v for k, v in results.items() if "_sim" not in k}

    if not filtered:
        print("No results found.")
        return

    # Sort by step name
    items = sorted(filtered.items())
    baseline = next((v for k, v in items if "baseline" in k), None)

    # Header
    print(f"\n{'Step':<25} {'RTF':>7} {'RTF(ASR)':>9} {'WER':>7} {'dWER':>7} {'GPU MB':>7} {'ASR(s)':>8} {'Align(s)':>9} {'Files':>5}")
    print("─" * 95)

    for name, data in items:
        rtf = data.get("rtf", 0)
        rtf_asr = data.get("rtf_asr_only", 0)
        wer = data.get("mean_wer", 0)
        gpu = data.get("gpu_peak_mb", 0)
        n = data.get("n_files", 0)

        # Compute ASR and align totals from per_file data
        per_file = data.get("per_file", [])
        asr_total = sum(f.get("t_asr", 0) for f in per_file)
        align_total = sum(f.get("t_align", 0) for f in per_file)

        # Delta WER vs baseline
        if baseline and name != baseline.get("config_name"):
            d_wer = wer - baseline.get("mean_wer", 0)
            d_wer_str = f"{d_wer:+.3f}"
        else:
            d_wer_str = "--"

        print(f"{name:<25} {rtf:>7.4f} {rtf_asr:>9.4f} {wer:>7.4f} {d_wer_str:>7} {gpu:>7.0f} {asr_total:>8.1f} {align_total:>9.1f} {n:>5}")


def print_simulation_table():
    """Print simulation results comparison."""
    results = load_latest_results()
    sim_results = {k: v for k, v in results.items() if "_sim" in k}

    if not sim_results:
        print("No simulation results found.")
        return

    items = sorted(sim_results.items())

    print(f"\n{'Config':<30} {'Calls':>5} {'Chunks':>7} {'Avg(s)':>7} {'P95(s)':>7} {'Max(s)':>7} {'Over15s':>8} {'Tput':>7} {'RTF':>7}")
    print("─" * 105)

    for name, data in items:
        n_calls = data.get("n_calls", 0)
        total_chunks = data.get("total_chunks", 0)
        avg = data.get("avg_latency", 0)
        p95 = data.get("p95_latency", 0)
        max_lat = data.get("max_latency", 0)
        over = data.get("chunks_over_budget", 0)
        tput = data.get("throughput_chunks_per_sec", 0)
        rtf = data.get("effective_rtf", 0)

        pct_over = f"{over}/{total_chunks}" if total_chunks else "0/0"
        print(f"{name:<30} {n_calls:>5} {total_chunks:>7} {avg:>7.3f} {p95:>7.3f} {max_lat:>7.3f} {pct_over:>8} {tput:>7.2f} {rtf:>7.4f}")


def print_quality_summary(step_names: list[str] | None = None):
    """Print quality metrics comparison."""
    results = load_latest_results()
    if step_names:
        filtered = {k: v for k, v in results.items() if any(s in k for s in step_names)}
    else:
        filtered = {k: v for k, v in results.items() if "_sim" not in k}

    if not filtered:
        print("No results found.")
        return

    items = sorted(filtered.items())
    metrics = ["avg_logprob", "no_speech_prob", "compression_ratio"]

    print(f"\n{'Step':<25}", end="")
    for m in metrics:
        print(f" {m:>18}", end="")
    print()
    print("─" * (25 + 19 * len(metrics)))

    for name, data in items:
        qs = data.get("quality_summary", {})
        print(f"{name:<25}", end="")
        for m in metrics:
            val = qs.get(m, {}).get("mean", 0)
            print(f" {val:>18.4f}", end="")
        print()
