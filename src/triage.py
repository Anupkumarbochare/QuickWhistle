"""
QuickWhistle — intake / triage (runs BEFORE retrieval in the answer flow).

Classifies an incoming message into one of:
    greeting | chitchat | rules_question | off_topic

Only `rules_question` triggers the retrieve -> ground -> cite pipeline. Greetings
and chitchat get a short friendly canned reply (no retrieval, no Sources block);
off-topic gets a polite redirect.

Design goal: cheap + deterministic. A fast rule-based check resolves the common
cases (a greeting like "hi", a thanks, or any message containing hockey-rule
vocabulary) with NO LLM call. Only genuinely ambiguous messages (no hockey
signal, not a greeting/chitchat) fall back to a one-shot LLM classification.

This lives entirely in the answer/app layer — the verbatim system prompt is
untouched.
"""

from __future__ import annotations

import re

GREETING = "greeting"
CHITCHAT = "chitchat"
RULES = "rules_question"
OFF_TOPIC = "off_topic"

# --- canned replies (no retrieval, no Sources) ---
GREETING_REPLY = (
    "Hey! I'm QuickWhistle — ask me about any ice-hockey rule across the NHL, "
    "PWHL, IIHF, AHL, NCAA, or USA Hockey and I'll give you a cited answer."
)
CHITCHAT_REPLY = (
    "Happy to help! I stick to ice-hockey rules though — ask me about icing, "
    "penalties, checking, offside, or anything in the NHL, PWHL, IIHF, AHL, "
    "NCAA, or USA Hockey rulebooks and I'll give you a cited answer."
)
OFF_TOPIC_REPLY = (
    "I'm QuickWhistle and I can only answer ice-hockey rules questions. Ask me "
    "about a rule in the NHL, PWHL, IIHF, AHL, NCAA, or USA Hockey and I'll give "
    "you a cited answer."
)

# --- rule-based vocabulary ---
_GREETING_TOKENS = {
    "hi", "hii", "hello", "hey", "heya", "hiya", "yo", "howdy", "sup",
    "greetings", "gm", "hai", "helloo",
}
_GREETING_PHRASES = ("good morning", "good afternoon", "good evening", "good day")

_CHITCHAT_PATTERNS = (
    "thank", "thanks", "thx", "ty ", "cheers",
    "how are you", "how're you", "how r u", "how are u", "hows it going",
    "how's it going", "what's up", "whats up", "wassup",
    "who are you", "what are you", "what can you do", "what do you do",
    "tell me about yourself", "are you a bot", "are you ai",
    "good bot", "nice", "cool", "awesome", "lol", "haha", "ok", "okay",
    "bye", "goodbye", "see ya", "see you", "good night",
)

# Hockey-rule signal: presence of any of these => treat as a rules_question
# (cheap, high-recall). Leagues + common rule vocabulary.
_HOCKEY_TERMS = (
    # leagues
    "nhl", "pwhl", "iihf", "ahl", "ncaa", "usa hockey", "usah",
    # the game / surface / objects
    "hockey", "puck", "rink", "crease", "net", "goal", "goalie", "goaltender",
    "goalkeeper", "stick", "blue line", "red line", "zone", "boards", "glass",
    # officials / flow
    "referee", "ref ", "linesman", "linesperson", "whistle", "faceoff",
    "face-off", "face off", "period", "overtime", "shootout", "intermission",
    "power play", "powerplay", "penalty kill", "icing the puck",
    # infractions / penalties
    "icing", "offside", "off-side", "off side", "penalt", "minor", "major",
    "misconduct", "checking", "body check", "bodycheck", "hit", "hitting",
    "fight", "fighting", "slash", "trip", "hook", "hold", "cross-check",
    "cross check", "high stick", "high-stick", "boarding", "charging",
    "interference", "delay of game", "too many men", "spearing", "elbow",
    "butt-end", "roughing", "diving", "embellish",
    # rules-ish / dimensions / tournament
    "rule", "rulebook", "dimension", "regulation", "infraction", "violation",
    "relegation", "seeding", "olympic", "world championship",
    # measurements / units / conversion (rink, goal, stick, puck sizes) — these
    # let a dimension follow-up like "how wide is that in meters?" run the
    # retrieve+convert path instead of being redirected as off-topic.
    "meter", "metre", "feet", "foot", "inch", "cm", "convert",
    "how wide", "how long", "how far", "how tall",
    # equipment / gear
    "equipment", "protective", "gear", "skate", "helmet", "pad", "visor",
    "jersey",
    # legality / permission phrasing ("is X legal / allowed / required?")
    "required", "legal", "allowed",
)


# The assistant's own name contains "whistle" (a hockey term), so strip it
# before signal detection — otherwise "hey QuickWhistle" looks like a rules query.
_BOT_NAMES = ("quickwhistle", "quick whistle", "quick-whistle")


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def _has_hockey_signal(low: str) -> bool:
    for name in _BOT_NAMES:
        low = low.replace(name, " ")
    return any(term in low for term in _HOCKEY_TERMS)


def _is_greeting(low: str, words: list[str]) -> bool:
    if any(p in low for p in _GREETING_PHRASES):
        return True
    # short message that starts with / consists of a greeting token
    if words and words[0] in _GREETING_TOKENS and len(words) <= 5:
        return True
    return False


def _is_chitchat(low: str, words: list[str]) -> bool:
    if len(words) > 8:  # long messages are unlikely to be pure chitchat
        return False
    return any(p in low for p in _CHITCHAT_PATTERNS)


# A short ambiguous message mid-conversation is almost always a rules
# follow-up ("how much is that?", "what about them?"), so we judge it in the
# context of the active session rather than in isolation.
_FOLLOWUP_MAX_WORDS = 8


def classify(message: str, llm_classify=None, has_session: bool = False) -> str:
    """Return greeting | chitchat | rules_question | off_topic.

    Rule-based first (no LLM). `llm_classify`, if provided, is a callable
    (message -> label) used only for ambiguous messages with no hockey signal.

    `has_session` is True when a prior rules turn has set a topic/league in the
    session. When True, a SHORT ambiguous follow-up defaults to rules_question
    (and skips the LLM tiebreak) — a bare "how much is that in feet?" is a
    continuation of the conversation, not a fresh off-topic query. Greetings and
    chitchat are still caught first, so "thanks!" mid-chat stays chitchat.
    """
    low = message.strip().lower()
    words = _tokens(low)

    if not words:
        return GREETING  # empty/whitespace -> treat as a greeting nudge

    # 1. Any hockey-rule signal -> rules_question (high recall, cheapest win).
    if _has_hockey_signal(low):
        return RULES

    # 2. Greeting / chitchat (closed sets, deterministic).
    if _is_greeting(low, words):
        return GREETING
    if _is_chitchat(low, words):
        return CHITCHAT

    # 3. Ambiguous (no hockey signal, not greeting/chitchat).
    # 3a. Session-aware: a short follow-up inside an active conversation is a
    #     rules continuation. Resolve it in context; no LLM call needed.
    if has_session and len(words) <= _FOLLOWUP_MAX_WORDS:
        return RULES

    # 3b. Otherwise the optional LLM tiebreak decides rules_question vs
    #     off_topic (the ONLY place an LLM call may happen). If no tiebreak is
    #     available or it fails, default to rules_question — safer to attempt an
    #     answer than to wrongly redirect a real (oddly-phrased) rules question.
    if llm_classify is not None:
        try:
            label = llm_classify(message)
            if label in (GREETING, CHITCHAT, RULES, OFF_TOPIC):
                return label
        except Exception:
            pass
    return RULES


def canned_reply(label: str) -> str:
    return {
        GREETING: GREETING_REPLY,
        CHITCHAT: CHITCHAT_REPLY,
        OFF_TOPIC: OFF_TOPIC_REPLY,
    }[label]


# Prompt for the optional LLM tiebreak (kept tiny to stay cheap).
LLM_CLASSIFY_PROMPT = (
    "Classify the user's message into exactly one of these labels:\n"
    "  greeting       - a hello/greeting\n"
    "  chitchat       - small talk, thanks, or asking about you\n"
    "  rules_question - a question about ICE HOCKEY rules\n"
    "  off_topic      - anything not about ice hockey rules\n"
    "Reply with ONLY the label.\n\nMessage: {message}"
)
