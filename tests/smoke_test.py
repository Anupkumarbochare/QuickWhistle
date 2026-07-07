"""
End-to-end smoke test — runs on MODEL_PROVIDER=mock so it never touches the
Gemini daily quota.

Drives a 3-turn conversation through the real Streamlit app (via AppTest):
  1. "What is icing in the NHL?"        -> NHL
  2. "what about in the Olympics?"      -> resolves to IIHF (memory carry-forward)
  3. "Is hitting in the PWHL the same as in the NHL?" -> NHL + PWHL (cross-league)

Asserts each assistant turn renders an answer + the "view retrieved chunks"
expander, and that league detection/memory behave. Also runs the converter
tool's offline self-checks.

Run:  MODEL_PROVIDER=mock python tests/smoke_test.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("MODEL_PROVIDER", "mock")
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

results: list[bool] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    results.append(ok)
    mark = "PASS ✅" if ok else "FAIL ❌"
    print(f"  [{mark}] {label}" + (f"  ({detail})" if detail else ""))


def chat_smoke() -> None:
    from streamlit.testing.v1 import AppTest

    print("\n" + "=" * 70)
    print("MULTI-TURN CHAT SMOKE (MODEL_PROVIDER=mock)")
    print("=" * 70)

    at = AppTest.from_file("app.py", default_timeout=120).run()
    check("App boots without exception", at.exception == [] or not at.exception)

    turns = [
        ("What is icing in the NHL?", {"NHL"}, "explicit"),
        ("what about in the Olympics?", {"IIHF"}, "implied"),
        ("Is hitting in the PWHL the same as in the NHL?", {"NHL", "PWHL"}, "explicit"),
    ]

    for i, (q, expect_leagues, expect_mode) in enumerate(turns, 1):
        at.chat_input[0].set_value(q).run()
        captions = [c.value for c in at.caption]
        meta_caps = [c for c in captions if c.startswith("league(s):")]
        latest = meta_caps[-1] if meta_caps else ""
        expanders = [e.label for e in at.expander if "retrieved chunks" in e.label]
        print(f"\n  Turn {i}: {q!r}")
        print(f"    meta: {latest}")
        print(f"    expander: {expanders[-1] if expanders else '(none)'}")

        check(f"Turn {i}: assistant answered (no exception)",
              not at.exception)
        check(f"Turn {i}: chunk expander rendered", bool(expanders))
        ok_lg = all(lg in latest for lg in expect_leagues)
        check(f"Turn {i}: detected {expect_leagues}", ok_lg, latest)
        check(f"Turn {i}: detection mode '{expect_mode}'",
              f"detection: {expect_mode}" in latest)

    # message history accumulated: 3 user + 3 assistant
    n_msgs = len(at.chat_message)
    check("Full history retained (6 messages)", n_msgs == 6, f"{n_msgs} messages")


def triage_smoke() -> None:
    from streamlit.testing.v1 import AppTest

    from src.triage import CHITCHAT_REPLY, GREETING_REPLY

    print("\n" + "=" * 70)
    print("INTAKE/TRIAGE SMOKE — greeting / chitchat skip retrieval")
    print("=" * 70)

    at = AppTest.from_file("app.py", default_timeout=120).run()
    rows = [
        # (message, expected detection mode, canned reply, expect retrieval?)
        ("hi", "greeting", GREETING_REPLY, False),
        ("thanks!", "chitchat", CHITCHAT_REPLY, False),
        ("What is icing in the NHL?", "explicit", None, True),
    ]
    prev_expanders = 0
    for msg, mode, canned, expect_retrieval in rows:
        at.chat_input[0].set_value(msg).run()
        caps = [c.value for c in at.caption if c.value.startswith("league(s):")]
        latest = caps[-1] if caps else ""
        exp_count = len([e for e in at.expander if "retrieved chunks" in e.label])
        retrieved = exp_count > prev_expanders
        prev_expanders = exp_count
        mds = [m.value for m in at.markdown]

        print(f"\n  {msg!r}")
        print(f"    meta: {latest}")
        print(f"    retrieved chunks this turn: {retrieved}")

        check(f"{msg!r}: detection '{mode}'", f"detection: {mode}" in latest, latest)
        check(f"{msg!r}: retrieval {'ON' if expect_retrieval else 'OFF'}",
              retrieved == expect_retrieval)
        if canned:  # greeting / chitchat -> canned reply, no Sources block
            shown = any(canned[:50] in m for m in mds)
            no_sources = all("Sources" not in m for m in mds if canned[:50] in m)
            check(f"{msg!r}: canned reply shown, no Sources block",
                  shown and no_sources)
        else:  # rules question still flows through normally
            check(f"{msg!r}: rules answer cites Sources",
                  any("Sources" in m for m in mds))


def tool_smoke() -> None:
    print("\n" + "=" * 70)
    print("CONVERTER TOOL — offline self-checks (no LLM)")
    print("=" * 70)
    import math

    from src.tools import convert_units

    cases = [
        ("30 m -> ft", convert_units(30, "meters", "feet")["result"], 98.4252),
        ("85 ft -> m", convert_units(85, "feet", "meters")["result"], 25.908),
        ("170 g -> oz", convert_units(170, "grams", "ounces")["result"], 5.9966),
    ]
    for label, got, exp in cases:
        check(f"convert_units {label}", math.isclose(got, exp, rel_tol=1e-3),
              f"{got}")
    print("\n  NOTE: the model-driven tool CALL (Gemini function calling) is "
          "verified in the Phase 9 live demo:\n        `python src/tools.py "
          "--live` -> tool_calls fire (26 m->85.30 ft, 30 m->98.43 ft).")
    print("        Tool-calling is Gemini-specific, so the mock chat path does "
          "not invoke it.")


def main() -> None:
    chat_smoke()
    triage_smoke()
    tool_smoke()
    print("\n" + "=" * 70)
    total, passed = len(results), sum(results)
    print(f"SMOKE RESULT: {passed}/{total} checks passed")
    print("=" * 70)
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
