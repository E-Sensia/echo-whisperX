"""CLI entrypoint for running benchmark steps.

Usage:
    python -m bench.run_step --step 0              # Run baseline
    python -m bench.run_step --step 1a --step 1b   # Run specific steps
    python -m bench.run_step --step all             # Run all steps 0-5
    python -m bench.run_step --simulate 1 5 10 20   # Run live simulation
    python -m bench.run_step --compare              # Compare all results
    python -m bench.run_step --compare-sim          # Compare simulation results
    python -m bench.run_step --cross-call 1 5 10 20  # Cross-call batching simulation
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(description="echo-whisperX benchmark runner")
    parser.add_argument(
        "--step", nargs="*", default=None,
        help="Step(s) to run: 0, 1a, 1b, 2, 3, 4a, 4b, 5, or 'all'",
    )
    parser.add_argument(
        "--simulate", nargs="*", type=int, default=None,
        help="Run live simulation with N concurrent calls (e.g., --simulate 1 5 10 20)",
    )
    parser.add_argument(
        "--sim-config", default="0",
        help="Step config to use for simulation (default: 0 = baseline)",
    )
    parser.add_argument(
        "--compare", action="store_true",
        help="Print comparison table of all results",
    )
    parser.add_argument(
        "--compare-sim", action="store_true",
        help="Print simulation results comparison",
    )
    parser.add_argument(
        "--quality", action="store_true",
        help="Print quality metrics summary",
    )
    parser.add_argument(
        "--cross-call", nargs="*", type=int, default=None,
        help="Run cross-call batching simulation (e.g., --cross-call 5 10 20 50)",
    )
    args = parser.parse_args()

    if args.compare:
        from bench.compare import print_comparison_table
        print_comparison_table()
        return

    if args.compare_sim:
        from bench.compare import print_simulation_table
        print_simulation_table()
        return

    if args.quality:
        from bench.compare import print_quality_summary
        print_quality_summary()
        return

    if args.cross_call is not None:
        from bench.config import STEPS
        from bench.simulate import run_cross_call_simulation

        config = STEPS.get(args.sim_config)
        if config is None:
            print(f"Unknown config: {args.sim_config}. Available: {list(STEPS.keys())}")
            sys.exit(1)

        n_calls_list = args.cross_call if args.cross_call else [1, 5, 10, 20, 50]
        run_cross_call_simulation(config, n_calls_list)
        return

    if args.simulate is not None:
        from bench.config import STEPS
        from bench.simulate import run_simulation

        config = STEPS.get(args.sim_config)
        if config is None:
            print(f"Unknown config: {args.sim_config}. Available: {list(STEPS.keys())}")
            sys.exit(1)

        n_calls_list = args.simulate if args.simulate else [1, 5, 10, 20]
        run_simulation(config, n_calls_list)
        return

    if args.step is not None:
        from bench.config import STEPS
        from bench.harness import run_benchmark

        if "all" in args.step:
            steps_to_run = ["0", "1a", "1b", "2", "3", "4a", "4b", "5"]
        else:
            steps_to_run = args.step

        for step_id in steps_to_run:
            config = STEPS.get(step_id)
            if config is None:
                print(f"Unknown step: {step_id}. Available: {list(STEPS.keys())}")
                continue
            run_benchmark(config)
        return

    parser.print_help()


if __name__ == "__main__":
    main()
