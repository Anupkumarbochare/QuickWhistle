"""
Phase 5 acceptance harness.

Runs the 5 brief use cases + an off-topic redirect + an empty/no-match question
through the full retrieve -> ground -> generate pipeline, printing each answer
with its detected league(s) and the chunks that grounded it.

Backend is whatever config.MODEL_PROVIDER points at (gemini / ollama / mock):
    MODEL_PROVIDER=mock   python tests/run_acceptance.py     # offline wiring check
    MODEL_PROVIDER=gemini python tests/run_acceptance.py     # real answers
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.answer import answer  # noqa: E402

CASES = [
    ("USE CASE 1 (novice, icing)",
     "What is icing and why did the ref blow the whistle?", None),
    ("USE CASE 2 (cross-league hitting)",
     "Is hitting in the PWHL the same as in the NHL?", None),
    ("USE CASE 3 (youth checking age)",
     "At what age is checking allowed in USA Hockey?", None),
    ("USE CASE 4 (definitional penalties)",
     "What's the difference between a minor, major, misconduct, and game misconduct?",
     ["NHL"]),
    ("USE CASE 5 (IIHF tournament)",
     "How does seeding and relegation work at the IIHF Worlds?", None),
    ("OFF-TOPIC (should redirect)",
     "What's the best pizza topping for game night?", None),
    ("EMPTY/NO-MATCH (should refuse)",
     "What does the rulebook say about ticket refund and parking policies?", None),
]


def main() -> None:
    import time

    # Gemini free tier allows ~5 requests/minute; space calls to stay under it
    # (the adapter also retries on 429 as a backstop). 0 for local/mock.
    spacing = 13 if config.MODEL_PROVIDER == "gemini" else 0

    print(f"### Backend: MODEL_PROVIDER={config.MODEL_PROVIDER} "
          f"(model={config.MODEL})\n")
    for n, (label, q, leagues) in enumerate(CASES):
        if n and spacing:
            time.sleep(spacing)
        print("=" * 78)
        print(f"{label}\nQ: {q}")
        print("=" * 78)
        res = answer(q, leagues=leagues)
        print(f"[detection_mode={res['detection_mode']} | "
              f"leagues={res['leagues']} | grounded={res['grounded']} | "
              f"chunks_used={len(res['chunks'])}]\n")
        print(res["answer"])
        if res["chunks"]:
            print("\n  grounded by:")
            for c in res["chunks"]:
                print(f"    - {c['league']} Rule {c['rule_number']} ({c['rule_name']})")
        print()


if __name__ == "__main__":
    main()
