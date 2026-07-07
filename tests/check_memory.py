"""
Phase 6 self-check — run me directly:

    .venv/bin/python tests/check_memory.py

Demonstrates two things end-to-end:

  (1) SESSION follow-up resolution
      - "what is icing?"            (no league -> ambiguous)
      - "what about in the Olympics?"  resolves to IIHF Rule 81 (Icing),
        with the model stating its assumption explicitly.
      (Uses the real Gemini backend from .env — 2 calls, lightly throttled.)

  (2) LONG-TERM opt-in lifecycle
      - writes are REFUSED while disabled
      - enable -> set -> persist -> reload-from-disk -> delete

Prints a PASS/FAIL line per check and exits non-zero if anything fails.
No LLM is needed for part (2).
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Quiet Chroma's telemetry noise before it imports.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.memory import (  # noqa: E402
    LongTermMemory,
    SessionMemory,
    answer_with_memory,
)

PASS, FAIL = "PASS ✅", "FAIL ❌"
results: list[bool] = []


def check(label: str, ok: bool) -> None:
    results.append(ok)
    print(f"  [{PASS if ok else FAIL}] {label}")


# ===========================================================================
def part1_session() -> None:
    print("\n" + "=" * 72)
    print("(1) SESSION MEMORY — ambiguous follow-up resolves to IIHF icing")
    print("=" * 72)

    sess = SessionMemory()

    print('\n--- Turn 1: "what is icing?" (no league named) ---')
    r1 = answer_with_memory("what is icing?", sess)
    print(f"  detection_mode = {r1['detection_mode']}  (expected: ambiguous)")
    print(f"  answer: {r1['answer'][:240].strip()}...")
    check("Turn 1 detected as ambiguous (no league named)",
          r1["detection_mode"] == "ambiguous")

    time.sleep(13)  # stay under the Gemini free-tier rate limit

    print('\n--- Turn 2: "what about in the Olympics?" (follow-up) ---')
    r2 = answer_with_memory("what about in the Olympics?", sess)
    print(f"  detection_mode = {r2['detection_mode']}  (expected: implied)")
    print(f"  leagues        = {r2['leagues']}  (expected: ['IIHF'])")
    cited = {(c["league"], c["rule_number"]) for c in r2["chunks"]}
    print(f"  grounded by    = {sorted(cited)}")
    print("\n  --- answer ---")
    print("  " + r2["answer"].replace("\n", "\n  "))

    check("Follow-up resolved to IIHF (Olympics -> IIHF gameplay)",
          r2["leagues"] == ["IIHF"])
    check("Grounded in IIHF Rule 81 (Icing)", ("IIHF", "81") in cited)
    lower = r2["answer"].lower()
    check("Model stated its assumption explicitly",
          ("i'll take that" in lower or "assum" in lower
           or ("iihf" in lower and "olympic" in lower)))


# ===========================================================================
def part2_longterm() -> None:
    print("\n" + "=" * 72)
    print("(2) LONG-TERM MEMORY — opt-in: refuse-while-disabled, then lifecycle")
    print("=" * 72)

    user = "check_memory_demo"
    path = config.USER_MEMORY_DIR / f"{user}.json"
    lt = LongTermMemory(user, path=path)
    lt.delete()  # start clean
    lt = LongTermMemory(user, path=path)

    print(f"\n  store path: {path}")
    check("Disabled by default", lt.is_enabled is False)

    # Writes refused while disabled.
    refused = False
    try:
        lt.set_preferred_league("IIHF")
    except RuntimeError:
        refused = True
    check("Write REFUSED while disabled", refused)
    check("Nothing persisted while disabled", not path.exists())

    # Enable -> set -> persist.
    lt.enable()
    lt.set_preferred_league("IIHF")
    lt.set_expertise("technical")
    check("File persisted after enable+set", path.exists())

    # Reload from disk in a fresh object.
    lt2 = LongTermMemory(user, path=path)
    d = lt2.as_defaults()
    print(f"  reloaded from disk: {d}")
    check("Reload preserves preferred_league=IIHF",
          d.get("preferred_league") == "IIHF")
    check("Reload preserves expertise=technical",
          d.get("expertise") == "technical")

    # Delete (right to be forgotten).
    lt2.delete()
    check("Delete removes the file", not path.exists())
    check("Delete resets to disabled defaults", lt2.as_defaults() == {})


# ===========================================================================
def main() -> None:
    print(f"### Backend: MODEL_PROVIDER={config.MODEL_PROVIDER} (model={config.MODEL})")
    part1_session()
    part2_longterm()

    print("\n" + "=" * 72)
    total, passed = len(results), sum(results)
    print(f"RESULT: {passed}/{total} checks passed")
    print("=" * 72)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
