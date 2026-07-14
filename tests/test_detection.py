"""
League-detection tests (Fix 2): new NHL aliases + AHL non-regression.

Deterministic and free — no LLM, no retrieval. Checks detect_leagues() maps the
new "North American"/"American"/"Canadian"/"US" phrasings to NHL, without
stealing the AHL's "American Hockey League" name.

Run:  python tests/test_detection.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.retrieve import detect_leagues  # noqa: E402

results: list[bool] = []


def check(msg: str, expect: set[str]) -> None:
    leagues, mode = detect_leagues(msg)
    ok = set(leagues) == expect
    results.append(ok)
    mark = "PASS ✅" if ok else "FAIL ❌"
    print(f"  [{mark}] {msg!r} -> {leagues} ({mode})  want {sorted(expect)}")


def main() -> None:
    print("=" * 64)
    print("LEAGUE DETECTION — new NHL aliases (Fix 2)")
    print("=" * 64)
    # New aliases resolve to NHL.
    check("how wide is a North American rink?", {"NHL"})
    check("what are the northamerican rink dimensions?", {"NHL"})
    check("is fighting allowed under american rules?", {"NHL"})
    check("what is icing in canadian hockey?", {"NHL"})
    check("how does offside work for us?", {"NHL"})

    # Non-regression: "American Hockey League" must stay AHL, not NHL.
    check("what is icing in the American Hockey League?", {"AHL"})
    # Existing explicit signals unchanged.
    check("What is icing in the NHL?", {"NHL"})
    check("PWHL vs NHL hitting", {"NHL", "PWHL"})

    total, passed = len(results), sum(results)
    print("-" * 64)
    print(f"DETECTION RESULT: {passed}/{total} passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
