"""Project test runner with concise, readable terminal output."""

import argparse
import io
import os
import sys
import time
import unittest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run test suite with terminal summary")
    parser.add_argument(
        "--no-coverage",
        action="store_true",
        help="Disable coverage calculation",
    )
    return parser.parse_args()


def _start_coverage(enabled: bool):
    if not enabled:
        return None, "disabled"

    try:
        from coverage import Coverage
    except Exception:
        return None, "missing"

    cwd = os.getcwd()
    cov = Coverage(
        source=[cwd],
        omit=[
            "*/tests/*",
            "*/.venv/*",
            "*/run_tests.py",
        ],
    )
    cov.start()
    return cov, "enabled"


def main() -> None:
    args = _parse_args()

    print("=" * 72)
    print("semantic-v2 test suite")
    print("discovering tests under tests/")
    print("=" * 72)

    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")

    coverage_engine, coverage_state = _start_coverage(enabled=not args.no_coverage)

    start = time.time()
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    elapsed = time.time() - start

    coverage_pct = None
    if coverage_engine is not None:
        coverage_engine.stop()
        coverage_engine.save()
        report_buffer = io.StringIO()
        coverage_pct = coverage_engine.report(
            file=report_buffer,
            skip_empty=True,
            show_missing=False,
        )

    print("-" * 72)
    print(f"tests run : {result.testsRun}")
    print(f"failures  : {len(result.failures)}")
    print(f"errors    : {len(result.errors)}")
    print(f"skipped   : {len(result.skipped)}")
    if coverage_pct is not None:
        print(f"coverage  : {coverage_pct:.2f}%")
    elif coverage_state == "missing":
        print("coverage  : unavailable (install with: python -m pip install coverage)")
    else:
        print("coverage  : disabled")
    print(f"duration  : {elapsed:.2f}s")
    print("status    : PASS" if result.wasSuccessful() else "status    : FAIL")
    print("=" * 72)

    sys.exit(0 if result.wasSuccessful() else 1)


if __name__ == "__main__":
    main()
