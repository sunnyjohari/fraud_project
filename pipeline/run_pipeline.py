import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import time


STAGES = [
    ("ingest",     "pipeline.ingest"),
    ("preprocess", "pipeline.preprocess"),
    ("train",      "pipeline.train"),
    ("evaluate",   "pipeline.evaluate"),
]


def run_stage(name, module_path):
    import importlib
    print(f"\n{'─' * 48}")
    print(f"  Stage: {name}")
    print(f"{'─' * 48}")
    t0 = time.perf_counter()
    mod = importlib.import_module(module_path)
    mod.run()
    elapsed = time.perf_counter() - t0
    print(f"  ✓  {name} completed in {elapsed:.2f}s")
    return elapsed


def main():
    print("=" * 48)
    print("  Fraud Detection Pipeline")
    print("=" * 48)

    total_start = time.perf_counter()
    timings = {}

    for name, module_path in STAGES:
        timings[name] = run_stage(name, module_path)

    total = time.perf_counter() - total_start
    print(f"\n{'=' * 48}")
    print(f"  Pipeline complete in {total:.2f}s")
    for name, t in timings.items():
        print(f"    {name:<14} {t:.2f}s")
    print("=" * 48)


if __name__ == "__main__":
    main()
