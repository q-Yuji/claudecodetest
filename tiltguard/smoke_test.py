"""TiltGuard smoke test: overlay opens on all monitors and closes itself.

Runs the overlay for 0.1 minutes (~6s) and asserts it returned on its own
within a sane window. Run: python tiltguard/smoke_test.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tiltguard.overlay import run_overlay  # noqa: E402


def main() -> int:
    start = time.monotonic()
    run_overlay(
        0.1,
        ["SMOKE TEST -- this overlay closes itself in ~6 seconds."],
        "unlock",
    )
    elapsed = time.monotonic() - start
    assert 4.0 <= elapsed <= 30.0, f"overlay closed after {elapsed:.1f}s, expected ~6s"
    print(f"smoke test passed: overlay self-closed after {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
