"""
Long-term memory field tests (Fix G): role_type + youth_division.

Verifies the opt-in lifecycle: enable -> set -> reload persists -> delete clears,
plus the opt-in guard (no writes while disabled) and value validation. Uses a
throwaway temp path so real user memory is untouched.

Run:  python tests/test_longterm_fields.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.memory import LongTermMemory  # noqa: E402

results: list[bool] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    results.append(ok)
    mark = "PASS ✅" if ok else "FAIL ❌"
    print(f"  [{mark}] {label}" + (f"  ({detail})" if detail else ""))


def main() -> None:
    print("=" * 60)
    print("LONG-TERM MEMORY — role_type + youth_division (Fix G)")
    print("=" * 60)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "u.json"

        # 1. Opt-in guard: setting before enable raises, nothing persisted.
        m = LongTermMemory("u", path=path)
        try:
            m.set_role_type("coach")
            check("set before enable raises", False)
        except RuntimeError:
            check("set before enable raises", True)
        check("nothing written while disabled", not path.exists())

        # 2. Enable -> set both new fields.
        m.enable()
        m.set_role_type("coach")
        m.set_youth_division("bantam")
        check("role_type set", m.data["role_type"] == "coach")
        check("youth_division set", m.data["youth_division"] == "bantam")

        # 3. Reload from disk -> values persist.
        m2 = LongTermMemory("u", path=path)
        check("reload persists role_type", m2.data["role_type"] == "coach", m2.data["role_type"])
        check("reload persists youth_division",
              m2.data["youth_division"] == "bantam", m2.data["youth_division"])
        check("as_defaults exposes new fields (enabled)",
              m2.as_defaults().get("role_type") == "coach"
              and m2.as_defaults().get("youth_division") == "bantam")

        # 4. Validation: bad values rejected.
        try:
            m2.set_role_type("wizard")
            check("invalid role_type rejected", False)
        except ValueError:
            check("invalid role_type rejected", True)
        try:
            m2.set_youth_division("college")
            check("invalid youth_division rejected", False)
        except ValueError:
            check("invalid youth_division rejected", True)

        # 5. Delete -> file gone, fields cleared to defaults.
        m2.delete()
        check("delete removes file", not path.exists())
        check("delete clears role_type", m2.data["role_type"] is None)
        check("delete clears youth_division", m2.data["youth_division"] is None)
        check("delete disables memory", m2.is_enabled is False)

    total, passed = len(results), sum(results)
    print("-" * 60)
    print(f"LONGTERM RESULT: {passed}/{total} passed")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
