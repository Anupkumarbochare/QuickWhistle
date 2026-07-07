"""
QuickWhistle — Phase 6: Memory.

Two layers:

1. SessionMemory (always on, in-RAM, single chat session)
   Tracks the last rule topic and last league(s) so an ambiguous follow-up like
   "what about in the Olympics?" can be resolved:
     * the previous question is fed to retrieval's query expansion (carries the
       topic, e.g. "icing", forward), and
     * the previous league(s) become the default when the new question names
       none (carry-forward) — though an explicit/implied new league always wins
       (e.g. "Olympics" -> IIHF), never silently assumed.
   No cross-session persistence (matches the system prompt).

2. LongTermMemory (OPT-IN, persisted per user as JSON)
   Stores a preferred league + expertise level across sessions. Off by default;
   nothing is written until enabled. Includes a clear on/off flag and a delete
   (right-to-be-forgotten) function.

Wiring: `answer_with_memory()` pulls context from memory, calls answer(), then
records the turn. answer() itself stays pure (Phase 5).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.answer import answer  # noqa: E402


# ===========================================================================
# Session memory (in-RAM, per chat session)
# ===========================================================================
class SessionMemory:
    """Last topic + last league(s) for follow-up resolution within a session."""

    def __init__(self) -> None:
        self.turns: list[dict] = []
        self.last_question: str | None = None
        self.last_leagues: list[str] = []
        self.last_topic: str | None = None  # e.g. the top retrieved rule_name

    def remember(
        self, question: str, leagues: list[str] | None, topic: str | None = None
    ) -> None:
        self.last_question = question
        # Only carry forward a *specific* league set (1-2 leagues). An ambiguous
        # turn that fanned out to all leagues shouldn't pin a default.
        if leagues and len(leagues) < len(config.LEAGUES):
            self.last_leagues = list(leagues)
        if topic:
            self.last_topic = topic
        self.turns.append(
            {"question": question, "leagues": list(leagues or []), "topic": topic}
        )

    def recall(self) -> dict:
        return {
            "prev_question": self.last_question,
            "default_leagues": list(self.last_leagues),
            "last_topic": self.last_topic,
        }

    def reset(self) -> None:
        self.__init__()


# ===========================================================================
# Long-term memory (opt-in, persisted)
# ===========================================================================
class LongTermMemory:
    """Per-user JSON store: preferred league + expertise. Opt-in; deletable."""

    DEFAULT = {"enabled": False, "preferred_league": None, "expertise": None}

    def __init__(self, user_id: str, path: Path | None = None) -> None:
        self.user_id = user_id
        self.path = path or (config.USER_MEMORY_DIR / f"{user_id}.json")
        self.data = self._load()

    def _load(self) -> dict:
        if self.path.exists():
            try:
                return {**self.DEFAULT, **json.loads(self.path.read_text())}
            except (json.JSONDecodeError, OSError):
                pass
        return dict(self.DEFAULT)

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf-8")

    # --- on/off ---
    @property
    def is_enabled(self) -> bool:
        return bool(self.data.get("enabled"))

    def enable(self) -> None:
        self.data["enabled"] = True
        self._save()

    def disable(self) -> None:
        """Turn off use of long-term memory but keep the stored values."""
        self.data["enabled"] = False
        self._save()

    # --- preferences (only persisted while enabled) ---
    def set_preferred_league(self, league: str | None) -> None:
        if not self.is_enabled:
            raise RuntimeError("Enable long-term memory before setting preferences.")
        self.data["preferred_league"] = league.upper() if league else None
        self._save()

    def set_expertise(self, level: str | None) -> None:
        if not self.is_enabled:
            raise RuntimeError("Enable long-term memory before setting preferences.")
        if level not in (None, "plain", "technical"):
            raise ValueError("expertise must be 'plain', 'technical', or None.")
        self.data["expertise"] = level
        self._save()

    # --- right to be forgotten ---
    def delete(self) -> None:
        """Erase the stored file and reset to defaults (off)."""
        if self.path.exists():
            self.path.unlink()
        self.data = dict(self.DEFAULT)

    def as_defaults(self) -> dict:
        """Preferences to apply, only when enabled; empty dict otherwise."""
        if not self.is_enabled:
            return {}
        return {
            "preferred_league": self.data.get("preferred_league"),
            "expertise": self.data.get("expertise"),
        }


# ===========================================================================
# Context resolution + memory-aware answering
# ===========================================================================
def resolve_context(
    session: SessionMemory, longterm: LongTermMemory | None = None
) -> tuple[str | None, list[str], str | None]:
    """Combine session + (opt-in) long-term memory into retrieval context.

    Returns (prev_question, default_leagues, expertise).
    Session leagues take precedence; long-term preferred league is the fallback
    default when the session has none.
    """
    rec = session.recall()
    prev_question = rec["prev_question"]
    default_leagues = rec["default_leagues"]
    expertise = None

    if longterm is not None:
        d = longterm.as_defaults()
        if d:
            if not default_leagues and d.get("preferred_league"):
                default_leagues = [d["preferred_league"]]
            expertise = d.get("expertise")

    return prev_question, default_leagues, expertise


def _session_context_line(
    session: SessionMemory, expertise: str | None
) -> str | None:
    """A short note for the LLM so it can resolve/echo ambiguous follow-ups."""
    rec = session.recall()
    if not rec["prev_question"]:
        return None
    parts = [
        f'Earlier this session the user asked: "{rec["prev_question"]}"'
    ]
    if rec["default_leagues"]:
        parts.append(f"(most recent league context: {', '.join(rec['default_leagues'])})")
    if rec["last_topic"]:
        parts.append(f"(most recent topic: {rec['last_topic']})")
    note = " ".join(parts)
    note += (
        " If the new question is an ambiguous follow-up, state your assumption "
        "explicitly before answering rather than asking the user to restate."
    )
    if expertise == "technical":
        note += " The user has indicated a preference for technical detail."
    return note


def answer_with_memory(
    question: str,
    session: SessionMemory,
    longterm: LongTermMemory | None = None,
    k: int = config.TOP_K,
) -> dict:
    """Memory-aware wrapper around answer(): resolves context, then records."""
    prev_question, default_leagues, expertise = resolve_context(session, longterm)
    session_context = _session_context_line(session, expertise)

    res = answer(
        question,
        k=k,
        prev_question=prev_question,
        default_leagues=default_leagues,
        session_context=session_context,
    )

    # Only record genuine rules turns. Greeting/chitchat/off_topic carry no
    # rule topic or league, so we don't let them overwrite the conversation
    # context used to resolve later follow-ups.
    if res["detection_mode"] not in ("greeting", "chitchat", "off_topic"):
        topic = res["chunks"][0]["rule_name"] if res["chunks"] else session.last_topic
        leagues_used = res["leagues"] or default_leagues
        session.remember(question, leagues_used, topic)
    return res
