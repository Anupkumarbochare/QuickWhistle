"""
QuickWhistle — Streamlit chat UI.

    streamlit run app.py

Features:
  * chat-style message history (st.chat_message / st.chat_input)
  * rendered grounded answer (the model's Sources block is part of it)
  * colored league / detection / grounded badges under each answer
  * a "🔎 view retrieved chunks" expander showing each chunk's league + rule
  * clickable example questions on the empty screen
  * wired to answer.py + memory.py (session memory carries follow-ups; opt-in
    long-term preferences live in the sidebar)
  * a visible "thinking…" spinner so latency never looks frozen
  * backend is swappable via MODEL_PROVIDER (anthropic / gemini / ollama / mock)
    — the UI runs against any of them unchanged.

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
# Client-side spacing between LLM calls on the rate-limited Gemini free tier. The
# adapter also retries on 429; this just smooths rapid-fire messages. No-op on
# Anthropic / Ollama / mock.
SPACING_SECONDS = 13 if config.MODEL_PROVIDER == "gemini" else 0

# Conversational modes never retrieve, so a "grounded" badge would be noise.
_CONVERSATIONAL = {"greeting", "chitchat", "off_topic"}

# Starter questions shown on the empty screen (each exercises a different path).
EXAMPLES = [
    "What is icing in the NHL?",
    "Is body checking allowed in the PWHL?",
    "How wide is an IIHF rink in feet?",
    "What's the penalty for fighting in the NHL?",
    "Difference between a minor and a major penalty?",
    "What does the NCAA say about checking from behind?",
]

_CSS = """
<style>
  .qw-pill { display:inline-block; padding:2px 10px; margin:2px 6px 2px 0;
             border-radius:999px; font-size:0.72rem; font-weight:600;
             line-height:1.6; white-space:nowrap; }
  .qw-league      { background:#E8F0FA; color:#1E5AA8; border:1px solid #cbdcf0; }
  .qw-mode        { background:#F1F5FA; color:#425466; border:1px solid #dde5ee; }
  .qw-grounded    { background:#E6F4EA; color:#1E7E34; border:1px solid #b7e0c2; }
  .qw-notgrounded { background:#FDECEC; color:#B02A2A; border:1px solid #f3c2c2; }
  .qw-tagline { color:#5b6b7b; font-size:0.98rem; margin:-6px 0 4px 0; }
  .qw-leaguerow { margin:2px 0 10px 0; }
</style>
"""


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
    """Sleep just enough to respect the Gemini free-tier limit (no-op otherwise)."""
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
    """Colored badges: detected league(s), how they were resolved, grounding."""
    if not meta:
        return
    mode = meta.get("detection_mode")
    pills = [
        f'<span class="qw-pill qw-league">{lg}</span>'
        for lg in (meta.get("leagues") or [])
    ]
    if mode:
        pills.append(f'<span class="qw-pill qw-mode">detection: {mode}</span>')
    # Grounding badge only where it's meaningful (skip greetings/chitchat/off-topic).
    if mode not in _CONVERSATIONAL:
        if meta.get("grounded"):
            pills.append('<span class="qw-pill qw-grounded">✓ grounded in rulebook</span>')
        else:
            pills.append('<span class="qw-pill qw-notgrounded">no matching rule</span>')
    st.markdown("".join(pills), unsafe_allow_html=True)


def render_message(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant":
            render_meta(msg.get("meta", {}))
            render_chunks_expander(msg.get("chunks", []))


def render_empty_state() -> None:
    """Welcome card + clickable example questions (shown when chat is empty)."""
    st.info(
        "👋 **Ask me anything about ice-hockey rules.** I pull the answer straight "
        "from the official rulebooks, explain it in plain language, and cite the "
        "exact rule. I can also convert rink/goal dimensions between metric and "
        "imperial."
    )
    st.markdown("###### Try one of these")
    cols = st.columns(2)
    for i, q in enumerate(EXAMPLES):
        if cols[i % 2].button(q, key=f"ex_{i}", use_container_width=True):
            st.session_state.pending_prompt = q
            st.rerun()


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

        roles = ["(none)"] + list(lt.ROLE_TYPES)
        cur_role = lt.data.get("role_type") or "(none)"
        sel_role = st.sidebar.selectbox(
            "Your role", roles,
            index=roles.index(cur_role) if cur_role in roles else 0,
        )
        lt.set_role_type(None if sel_role == "(none)" else sel_role)

        divs = ["(none)"] + list(lt.YOUTH_DIVISIONS)
        cur_div = lt.data.get("youth_division") or "(none)"
        sel_div = st.sidebar.selectbox(
            "Youth division", divs,
            index=divs.index(cur_div) if cur_div in divs else 0,
        )
        lt.set_youth_division(None if sel_div == "(none)" else sel_div)

        if st.sidebar.button("🗑️ Delete my stored data"):
            lt.delete()
            st.sidebar.success("Stored preferences deleted.")
            st.rerun()

    st.sidebar.divider()
    if st.sidebar.button("🧹 Clear chat & session memory"):
        st.session_state.messages = []
        st.session_state.session_memory = SessionMemory()
        st.session_state.pop("pending_prompt", None)
        st.rerun()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="QuickWhistle", page_icon="🏒")
    init_state()
    st.markdown(_CSS, unsafe_allow_html=True)
    render_sidebar()

    # --- Header ---
    st.title("QuickWhistle 🏒")
    st.markdown(
        '<div class="qw-tagline">Grounded, cited answers on ice-hockey rules — '
        "straight from the official rulebooks.</div>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="qw-leaguerow">'
        + "".join(f'<span class="qw-pill qw-league">{lg}</span>' for lg in config.LEAGUES)
        + "</div>",
        unsafe_allow_html=True,
    )

    # Replay history.
    for msg in st.session_state.messages:
        render_message(msg)

    # A typed message OR a clicked example question.
    typed = st.chat_input("Ask a hockey rules question…")
    prompt = typed or st.session_state.pop("pending_prompt", None)

    # Empty screen (nothing to process yet): welcome + example buttons.
    if not st.session_state.messages and not prompt:
        render_empty_state()
        return
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
            with st.spinner("QuickWhistle is checking the rulebooks…"):
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
