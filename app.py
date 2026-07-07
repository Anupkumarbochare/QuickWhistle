"""
QuickWhistle — Phase 7: Streamlit chat UI.

    streamlit run app.py

Features:
  * chat-style message history (st.chat_message / st.chat_input)
  * rendered grounded answer (the model's Sources block is part of it)
  * a structured "Sources" recap + a "🔎 view retrieved chunks" expander that
    shows each chunk's league + rule number (not just the text)
  * wired to answer.py + memory.py (session memory carries follow-ups; opt-in
    long-term preferences live in the sidebar)
  * a visible "thinking…" spinner so free-tier latency never looks frozen
  * 429 retry (in the Gemini adapter) + client-side request spacing here
  * backend is swappable via MODEL_PROVIDER (gemini / ollama / mock) — the UI
    runs against the offline stub unchanged.

Cross-platform: launch the same way on macOS and Windows; only the venv
activation path differs (see README).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from src.memory import LongTermMemory, SessionMemory, answer_with_memory  # noqa: E402

LOCAL_USER = "local_user"
# Client-side spacing between LLM calls on the rate-limited free tier. The
# adapter also retries on 429; this just smooths rapid-fire messages.
SPACING_SECONDS = 13 if config.MODEL_PROVIDER == "gemini" else 0


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
def init_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []          # [{role, content, chunks, meta}]
    if "session_memory" not in st.session_state:
        st.session_state.session_memory = SessionMemory()
    if "longterm" not in st.session_state:
        st.session_state.longterm = LongTermMemory(LOCAL_USER)
    if "last_llm_ts" not in st.session_state:
        st.session_state.last_llm_ts = 0.0


def throttle() -> None:
    """Sleep just enough to respect the free-tier rate limit (no-op if idle)."""
    if not SPACING_SECONDS:
        return
    elapsed = time.time() - st.session_state.last_llm_ts
    wait = SPACING_SECONDS - elapsed
    if wait > 0:
        with st.spinner(f"Respecting free-tier rate limit… ({int(wait)}s)"):
            time.sleep(wait)


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------
def render_chunks_expander(chunks: list[dict]) -> None:
    """Show retrieved chunks with league + rule number (the brief's requirement)."""
    if not chunks:
        return
    leagues = ", ".join(sorted({c["league"] for c in chunks}))
    with st.expander(f"🔎 View retrieved chunks ({len(chunks)} — {leagues})"):
        for i, c in enumerate(chunks, 1):
            retr = "+".join(c.get("retrievers", []))
            st.markdown(
                f"**{i}. {c['league']} · Rule {c['rule_number']} "
                f"({c['rule_name']})**  \n"
                f"*{c['section_header']}*  \n"
                f"<small>retrievers: {retr} · score {c.get('score')}</small>",
                unsafe_allow_html=True,
            )
            st.text(c["text"].strip())
            if i < len(chunks):
                st.divider()


def render_meta(meta: dict) -> None:
    """Small caption: detected league(s) + how they were resolved."""
    if not meta:
        return
    leagues = ", ".join(meta.get("leagues") or []) or "—"
    st.caption(
        f"league(s): {leagues} · detection: {meta.get('detection_mode')} · "
        f"grounded: {meta.get('grounded')}"
    )


def render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_meta(msg.get("meta", {}))
            render_chunks_expander(msg.get("chunks", []))


# ---------------------------------------------------------------------------
# Sidebar (backend info + opt-in long-term memory)
# ---------------------------------------------------------------------------
def render_sidebar() -> None:
    lt: LongTermMemory = st.session_state.longterm
    st.sidebar.title("QuickWhistle 🏒")
    st.sidebar.caption(
        f"Backend: **{config.MODEL_PROVIDER}** (`{config.MODEL}`)\n\n"
        f"Leagues: {', '.join(config.LEAGUES)}"
    )

    st.sidebar.subheader("Long-term memory (opt-in)")
    st.sidebar.caption(
        "Off by default. When on, your preferred league + expertise persist "
        "across sessions in a local file."
    )
    enabled = st.sidebar.checkbox("Remember my preferences", value=lt.is_enabled)
    if enabled and not lt.is_enabled:
        lt.enable()
    elif not enabled and lt.is_enabled:
        lt.disable()

    if lt.is_enabled:
        leagues = ["(none)"] + list(config.LEAGUES)
        cur_league = lt.data.get("preferred_league") or "(none)"
        sel = st.sidebar.selectbox(
            "Preferred league", leagues, index=leagues.index(cur_league)
            if cur_league in leagues else 0,
        )
        lt.set_preferred_league(None if sel == "(none)" else sel)

        exps = ["(infer automatically)", "plain", "technical"]
        cur_exp = lt.data.get("expertise") or "(infer automatically)"
        sel_exp = st.sidebar.selectbox(
            "Expertise level", exps,
            index=exps.index(cur_exp) if cur_exp in exps else 0,
        )
        lt.set_expertise(None if sel_exp.startswith("(infer") else sel_exp)

        if st.sidebar.button("🗑️ Delete my stored data"):
            lt.delete()
            st.sidebar.success("Stored preferences deleted.")
            st.rerun()

    st.sidebar.divider()
    if st.sidebar.button("🧹 Clear chat & session memory"):
        st.session_state.messages = []
        st.session_state.session_memory = SessionMemory()
        st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="QuickWhistle", page_icon="🏒")
    init_state()
    render_sidebar()

    st.title("QuickWhistle 🏒")
    st.caption(
        "Ask about ice hockey **rules** across the NHL, PWHL, IIHF, AHL, NCAA, "
        "and USA Hockey. Every answer is grounded in the rulebooks and cited."
    )

    # Replay history.
    for msg in st.session_state.messages:
        render_message(msg)

    prompt = st.chat_input("Ask a hockey rules question…")
    if not prompt:
        return

    # Show + store the user's turn.
    user_msg = {"role": "user", "content": prompt}
    st.session_state.messages.append(user_msg)
    render_message(user_msg)

    # Generate (memory-aware), with a visible thinking state.
    with st.chat_message("assistant"):
        try:
            throttle()
            with st.spinner(
                "QuickWhistle is thinking… (free-tier responses can take a "
                "few seconds)"
            ):
                res = answer_with_memory(
                    prompt,
                    st.session_state.session_memory,
                    st.session_state.longterm,
                )
            st.session_state.last_llm_ts = time.time()
        except Exception as e:  # surface backend errors instead of crashing
            err = (
                f"⚠️ The model backend ({config.MODEL_PROVIDER}) returned an "
                f"error:\n\n```\n{e}\n```\n\n"
                "Check your `.env` (API key / quota), or set `MODEL_PROVIDER="
                "mock` to try the UI offline."
            )
            st.markdown(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
            return

        st.markdown(res["answer"])
        meta = {
            "leagues": res["leagues"],
            "detection_mode": res["detection_mode"],
            "grounded": res["grounded"],
        }
        render_meta(meta)
        render_chunks_expander(res["chunks"])

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": res["answer"],
            "chunks": res["chunks"],
            "meta": meta,
        }
    )


if __name__ == "__main__":
    main()
