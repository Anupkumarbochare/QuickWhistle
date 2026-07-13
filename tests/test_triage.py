"""
Triage classification tests (Fix A + B).

Deterministic and free — no live LLM. The ambiguous-message tiebreak is
stubbed so the pizza case is repeatable; the LIVE "conversion tool fires on a
mid-conversation dimension follow-up" check lives in the smoke test.

Run:  python tests/test_triage.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import triage  # noqa: E402


def _stub_llm(message: str) -> str:
    """Stand-in for the LLM tiebreak. Anything reaching this point already has
    NO hockey signal and no active session, so a conservative classifier calls
    it off_topic. This lets us prove the session-aware short-circuit (Fix B)
    changes the outcome for the *same* bare follow-up."""
    return triage.OFF_TOPIC


# (message, has_session, expected_label)
CASES = [
    # --- the four the brief requires ---
    ("how much is this in meters?", True, triage.RULES),      # unit signal (mid-convo)
    ("what protective equipment is required", False, triage.RULES),  # equipment vocab
    ("convert that to feet", False, triage.RULES),            # convert/feet vocab
    ("what's a good pizza place?", False, triage.OFF_TOPIC),  # still redirected

    # --- session-awareness (Fix B): a bare follow-up with NO hockey signal ---
    ("how much is that?", True, triage.RULES),    # active session -> rules follow-up
    ("how much is that?", False, triage.OFF_TOPIC),  # no session -> tiebreak -> off_topic

    # --- greeting/chitchat still win even mid-conversation ---
    ("thanks!", True, triage.CHITCHAT),
    ("hi", True, triage.GREETING),

    # --- pizza mid-session is short, but "pizza" is not a hockey follow-up;
    #     session-awareness intentionally biases toward rules here (documented
    #     trade-off: judge follow-ups in context). Not asserted as off_topic.
]


def main() -> None:
    print("=" * 60)
    print("TRIAGE TESTS (Fix A + B)")
    print("=" * 60)
    passed = 0
    for msg, has_session, expected in CASES:
        got = triage.classify(msg, llm_classify=_stub_llm, has_session=has_session)
        ok = got == expected
        passed += ok
        mark = "PASS ✅" if ok else "FAIL ❌"
        sess = "session" if has_session else "no-session"
        print(f"  [{mark}] ({sess:10}) {msg!r} -> {got} (want {expected})")
    total = len(CASES)
    print("-" * 60)
    print(f"TRIAGE RESULT: {passed}/{total} passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
