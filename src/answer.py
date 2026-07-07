"""
QuickWhistle — Phase 5: Generation.

    question
      -> retrieve() hybrid chunks (Phase 4)
      -> relevance gate (poor matches -> empty context -> graceful refusal)
      -> format chunks into a <retrieved_context> block (with metadata)
      -> system prompt (verbatim) + context + question  ->  LLM adapter
      -> grounded, cited answer

The LLM is swappable via a single config value (config.MODEL_PROVIDER):
  * "gemini" - gemini-1.5-flash by default (free tier; needs GEMINI_API_KEY)
  * "ollama" - local llama3.1/llama3.2 (no API key)
  * "mock"   - offline, deterministic; no model. Builds a templated grounded
               answer from the retrieved chunks so the pipeline/formatting and
               the empty-retrieval refusal can be exercised without a backend.

Public entry point:
    answer(question, leagues=None, k=5, prev_question=None,
           default_leagues=None) -> dict
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src import triage  # noqa: E402
from src.retrieve import retrieve  # noqa: E402


# ===========================================================================
# Intake / triage helper
# ===========================================================================
def _llm_classify(message: str) -> str:
    """One-shot LLM label for ambiguous messages (the only triage LLM call).

    Uses the active backend. On the mock backend this returns a non-label
    string, so triage falls back to its safe default (rules_question).
    """
    out = get_llm().generate(
        "You are a terse text classifier. Reply with one word only.",
        triage.LLM_CLASSIFY_PROMPT.format(message=message),
    )
    return out.strip().split()[0].lower() if out.strip() else ""


# ===========================================================================
# Prompt + context assembly
# ===========================================================================
def load_system_prompt() -> str:
    """Load prompts/system_prompt.txt verbatim (no edits, no summarizing)."""
    return config.SYSTEM_PROMPT_PATH.read_text(encoding="utf-8")


def format_context(chunks: list[dict]) -> str:
    """Render retrieved chunks into a <retrieved_context> block.

    Each chunk exposes its metadata as attributes so the model can read the
    league / rule_number / rule_name / section_header for citation, exactly as
    the system prompt describes. An empty list yields an empty block, which the
    system prompt treats as "no relevant content retrieved".
    """
    if not chunks:
        return "<retrieved_context>\n(no relevant rulebook content was retrieved)\n</retrieved_context>"

    parts = ["<retrieved_context>"]
    for i, c in enumerate(chunks, 1):
        parts.append(
            f'  <chunk id="{i}" league="{c["league"]}" '
            f'rule_number="{c["rule_number"]}" '
            f'rule_name="{c["rule_name"]}" '
            f'section_header="{c["section_header"]}">'
        )
        parts.append(c["text"].strip())
        parts.append("  </chunk>")
    parts.append("</retrieved_context>")
    return "\n".join(parts)


def build_user_message(
    question: str, context: str, session_context: str | None = None
) -> str:
    """Combine retrieved context, optional session memory, and the question."""
    preamble = f"[Session context] {session_context}\n\n" if session_context else ""
    return (
        f"{preamble}{context}\n\n"
        f"Using only the retrieved rulebook content above, answer the user's "
        f"question.\n\nUser question: {question}"
    )


# ===========================================================================
# Relevance gate
# ===========================================================================
def is_relevant(chunks: list[dict]) -> bool:
    """True if at least one chunk is a strong dense match (see config)."""
    dists = [
        c["dense_distance"] for c in chunks if c.get("dense_distance") is not None
    ]
    if not dists:
        return False
    return min(dists) <= config.RELEVANCE_MAX_DISTANCE


# ===========================================================================
# LLM adapters (swappable)
# ===========================================================================
def _retry_delay_seconds(exc, default: int) -> int:
    """Pull the server-suggested retry delay out of a 429, else use default."""
    try:
        for d in getattr(exc, "details", lambda: [])():
            secs = getattr(getattr(d, "retry_delay", None), "seconds", 0)
            if secs:
                return secs + 1
    except Exception:
        pass
    return default


class GeminiAdapter:
    """Google Gemini (default). System prompt goes in as system_instruction."""

    def __init__(self) -> None:
        import google.generativeai as genai

        if not config.GEMINI_API_KEY:
            raise RuntimeError(
                "GEMINI_API_KEY is not set. Add it to .env, or set "
                "MODEL_PROVIDER=ollama / =mock."
            )
        genai.configure(api_key=config.GEMINI_API_KEY)
        self._genai = genai

    def generate(self, system_prompt: str, user_message: str) -> str:
        import time

        from google.api_core import exceptions as gexc

        model = self._genai.GenerativeModel(
            config.GEMINI_MODEL, system_instruction=system_prompt
        )
        # Free tier is rate-limited (a few requests/minute). Retry on 429 with
        # the server-suggested delay (falling back to a short backoff).
        for attempt in range(5):
            try:
                resp = model.generate_content(
                    user_message,
                    generation_config={
                        "temperature": config.TEMPERATURE,
                        "max_output_tokens": config.MAX_OUTPUT_TOKENS,
                    },
                )
                return resp.text.strip()
            except gexc.ResourceExhausted as e:
                wait = _retry_delay_seconds(e, default=8 * (attempt + 1))
                if attempt == 4:
                    raise
                print(f"  [gemini rate-limited; retrying in {wait}s]")
                time.sleep(wait)
        raise RuntimeError("unreachable")


class OllamaAdapter:
    """Local Ollama (no API key). System prompt is the system message."""

    def __init__(self) -> None:
        import ollama

        self._client = ollama.Client(host=config.OLLAMA_HOST)

    def generate(self, system_prompt: str, user_message: str) -> str:
        resp = self._client.chat(
            model=config.OLLAMA_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            options={"temperature": config.TEMPERATURE},
        )
        return resp["message"]["content"].strip()


class MockAdapter:
    """Offline, deterministic. NOT a real LLM — for pipeline/formatting tests.

    Produces a grounded, cited stub from the retrieved chunks, and emits the
    system prompt's exact graceful refusal when no context is present. This lets
    the acceptance harness verify wiring without a model backend.
    """

    REFUSAL = (
        "I wasn't able to find a matching rule in the retrieved content. "
        "You may want to consult the relevant league rulebook directly."
    )

    def generate(self, system_prompt: str, user_message: str) -> str:
        # Detect the empty-context case from the assembled user message.
        if "(no relevant rulebook content was retrieved)" in user_message:
            return self.REFUSAL
        # Parse chunk headers back out for a deterministic Sources block.
        import re

        chunks = re.findall(
            r'league="([^"]+)"\s+rule_number="([^"]+)"\s+rule_name="([^"]+)"',
            user_message,
        )
        seen, sources = set(), []
        for league, num, name in chunks:
            key = (league, num)
            if key in seen:
                continue
            seen.add(key)
            display = config.LEAGUE_DISPLAY.get(league, league)
            sources.append(f"- {display}, Rule {num} ({name})")
        body = (
            "[MOCK ADAPTER — no LLM configured] Based on the retrieved rulebook "
            "content, here is a grounded placeholder answer. Set MODEL_PROVIDER "
            "to gemini or ollama for a real generated answer."
        )
        return f"{body}\n\nSources:\n" + "\n".join(sources[:4])


def get_llm():
    """Return the adapter for config.MODEL_PROVIDER."""
    provider = config.MODEL_PROVIDER
    if provider == "gemini":
        return GeminiAdapter()
    if provider == "ollama":
        return OllamaAdapter()
    if provider == "mock":
        return MockAdapter()
    raise ValueError(
        f"Unknown MODEL_PROVIDER {provider!r}. Use gemini, ollama, or mock."
    )


# ===========================================================================
# Public API
# ===========================================================================
def answer(
    question: str,
    leagues: list[str] | None = None,
    k: int = config.TOP_K,
    *,
    prev_question: str | None = None,
    default_leagues: list[str] | None = None,
    session_context: str | None = None,
) -> dict:
    """Retrieve, ground, and generate a cited answer.

    Returns:
        {
          "answer": str,            # the model's grounded answer (or refusal)
          "chunks": list[dict],     # the chunks actually used (post-gate)
          "leagues": list[str],     # detected/used leagues
          "detection_mode": str,    # explicit / implied / greeting / chitchat / ...
          "grounded": bool,         # False when not a grounded rules answer
        }
    """
    # --- Intake/triage (before retrieval). Only rules_question proceeds to the
    # retrieve -> ground -> cite pipeline. greeting/chitchat/off_topic get a
    # cheap canned reply (no retrieval, no Sources). The LLM tiebreak is only
    # consulted for ambiguous, hockey-signal-free messages.
    label = triage.classify(question, llm_classify=_llm_classify)
    if label != triage.RULES:
        return {
            "answer": triage.canned_reply(label),
            "chunks": [],
            "leagues": [],
            "detection_mode": label,
            "grounded": False,
        }

    chunks = retrieve(
        question,
        leagues=leagues,
        k=k,
        prev_question=prev_question,
        default_leagues=default_leagues,
    )
    mode = chunks[0]["detection_mode"] if chunks else "ambiguous"
    used_leagues = sorted({c["league"] for c in chunks})

    # Relevance gate (safety net): only drops context on degenerate, very-far
    # retrieval. The PRIMARY refusal/redirect behavior is the LLM's, driven by
    # the verbatim system prompt — dense distance can't reliably tell off-topic
    # from no-match (both land ~0.4-0.5 with bge-small).
    grounded = is_relevant(chunks)
    context_chunks = chunks if grounded else []

    system_prompt = load_system_prompt()
    context = format_context(context_chunks)
    user_message = build_user_message(question, context, session_context)

    llm = get_llm()
    text = llm.generate(system_prompt, user_message)

    return {
        "answer": text,
        "chunks": context_chunks,
        "leagues": used_leagues if grounded else [],
        "detection_mode": mode,
        "grounded": grounded,
    }


if __name__ == "__main__":
    # Tiny manual check: print the assembled prompt for one question (no LLM).
    import json

    q = "What is icing in the NHL?"
    ch = retrieve(q, k=3)
    print(build_user_message(q, format_context(ch))[:1200])
    print("\n--- relevance gate:", is_relevant(ch))
