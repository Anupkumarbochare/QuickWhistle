"""
QuickWhistle — Phase 4: Hybrid retrieval.

    question
      -> detect target league(s)            (explicit / implied / ambiguous)
      -> per league: BM25 (keyword)  +  dense (BGE/Chroma)  candidates
      -> fuse with weighted Reciprocal Rank Fusion (RRF)
      -> metadata filter is inherent: one collection per league
      -> balanced merge across leagues -> top-k chunks (with metadata)

Public entry point (signature fixed by the brief):

    retrieve(question, leagues=None, k=5) -> list[dict]

Each returned chunk is the same dict shape Phase 5 will inject into the
<retrieved_context> block:

    {
      "league", "rule_number", "rule_name", "section_header", "text",
      "score",        # fused RRF score (higher = better)
      "retrievers",   # which methods surfaced it: e.g. ["dense", "bm25"]
    }

Optional keyword args (used by memory/UI in later phases, defaults keep the
brief's three-arg call working):
    prev_question   - previous user turn, for light follow-up query expansion
    default_leagues - league(s) carried forward from session memory
"""

from __future__ import annotations

import pickle
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

# Reuse the embedder, Chroma client, and tokenizer from the index builder so
# indexing and querying stay perfectly consistent.
from src.build_index import get_client, get_embedder, tokenize  # noqa: E402


# ===========================================================================
# League detection
# ===========================================================================
# Explicit signals: the league is named (or unambiguously referenced).
LEAGUE_ALIASES: dict[str, list[str]] = {
    "NHL": [r"\bnhl\b", r"national hockey league"],
    "PWHL": [r"\bpwhl\b", r"professional women'?s hockey league"],
    "IIHF": [
        r"\biihf\b",
        r"international ice hockey",
        r"world championship",
        r"\bworlds\b",
    ],
    "AHL": [r"\bahl\b", r"american hockey league"],
    "NCAA": [r"\bncaa\b", r"college hockey", r"collegiate"],
    "USA_HOCKEY": [r"\busa\s*hockey\b", r"\busah\b"],
}

# Implied signals: no league named, but the topic strongly points to one.
# Only consulted when there are no explicit hits.
LEAGUE_IMPLIED: dict[str, list[str]] = {
    # Youth/age-graded rules live in the USA Hockey book.
    "USA_HOCKEY": [
        r"\byouth\b", r"\bmite\b", r"\bsquirt\b", r"\bpee ?wee\b", r"\bbantam\b",
        r"\bage\b", r"\bu\d{1,2}\b", r"\bminor hockey\b", r"\bjunior\b",
    ],
    # Olympic *gameplay* is played under IIHF rules (see domain notes).
    "IIHF": [r"\bolympics?\b"],
    # Women's hockey (without naming PWHL) implies the PWHL book.
    "PWHL": [r"\bwomen'?s?\b", r"\bfemale\b", r"\bgirls\b"],
    # College references imply NCAA.
    "NCAA": [r"\buniversity\b"],
}


def detect_leagues(
    question: str, default_leagues: list[str] | None = None
) -> tuple[list[str], str]:
    """Return (leagues, mode).

    mode is one of:
      "explicit"      - one or more leagues named in the question
      "implied"       - inferred from topic keywords (no league named)
      "carry_forward" - none found, fell back to session default_leagues
      "ambiguous"     - none found and no default; returns ALL leagues so the
                        caller / system prompt can decide to clarify rather
                        than silently committing to a single guess
    """
    q = question.lower()

    explicit = [
        lg for lg, pats in LEAGUE_ALIASES.items()
        if any(re.search(p, q) for p in pats)
    ]
    implied = [
        lg for lg, pats in LEAGUE_IMPLIED.items()
        if any(re.search(p, q) for p in pats)
    ]

    # Collect ALL league matches in the query, not just the first kind found.
    # A comparison like "NHL vs the Olympics" names NHL explicitly and implies
    # IIHF (Olympics) — both must come back so cross-league answers aren't
    # silently narrowed to one league. Explicit hits set the mode; implied ones
    # are unioned in (an implied league already named explicitly isn't double
    # counted by _ordered).
    if explicit:
        return _ordered(explicit + implied), "explicit"
    if implied:
        return _ordered(implied), "implied"

    if default_leagues:
        return _ordered(default_leagues), "carry_forward"

    return list(config.LEAGUES), "ambiguous"


def _ordered(leagues) -> list[str]:
    """De-duplicate while keeping the canonical config.LEAGUES order."""
    s = set(leagues)
    return [lg for lg in config.LEAGUES if lg in s]


# ===========================================================================
# Follow-up query expansion
# ===========================================================================
# Function/question words ignored when testing topical overlap between turns.
_STOPWORDS = {
    "a", "an", "and", "the", "is", "are", "was", "were", "be", "been", "being",
    "of", "in", "on", "at", "to", "for", "from", "with", "by", "about", "as",
    "it", "its", "this", "that", "these", "those", "what", "whats", "how",
    "why", "when", "where", "which", "who", "do", "does", "did", "done", "if",
    "or", "so", "i", "me", "my", "we", "our", "you", "your", "he", "she",
    "they", "them", "explain", "tell", "please", "can", "could", "would",
    "should", "same", "there", "here", "any", "some", "just", "also", "no",
    "not", "get", "got", "give", "know", "mean", "means",
}
# League / venue references carry no rules topic of their own. A follow-up made
# up ONLY of these (e.g. "what about in the Olympics?") is a bare contextual
# reference and must inherit the previous turn's topic to retrieve well.
_CONTEXT_TOKENS = {
    "nhl", "pwhl", "iihf", "ahl", "ncaa", "usa", "usah", "hockey",
    "olympic", "olympics", "worlds", "world", "international",
    "europe", "european", "college", "collegiate", "university",
    "junior", "youth", "pro", "professional", "women", "womens", "womans",
    "league", "leagues",
}


def _content_tokens(text: str) -> set[str]:
    """Meaningful (non-stopword) tokens of a message."""
    return {t for t in tokenize(text) if t not in _STOPWORDS}


def expand_query(question: str, prev_question: str | None) -> str:
    """Prepend the previous question ONLY for genuine follow-ups.

    Bug fix: the old rule expanded any message <=6 words, so an unrelated new
    topic ("how wide are European rinks?") wrongly inherited the prior topic's
    chunks. Now we expand only when the follow-up is topically tied to the
    previous turn:
      1. it shares a non-stopword token with the previous question
         ("...icing" -> "...no touch icing"), OR
      2. it carries no rules topic of its own — only a league/venue reference
         ("what about in the Olympics?") — so it must inherit the prior topic.
    Otherwise it's treated as a fresh question and left untouched.
    """
    if not prev_question:
        return question

    cur = _content_tokens(question)
    prev = _content_tokens(prev_question)

    shares_topic = bool(cur & prev)
    residual = cur - _CONTEXT_TOKENS
    bare_context_ref = (not residual) and bool(cur & _CONTEXT_TOKENS)

    if shares_topic or bare_context_ref:
        return f"{prev_question.strip()} {question.strip()}"
    return question


# ===========================================================================
# Per-league candidate retrieval
# ===========================================================================
_BM25_CACHE: dict[str, dict] = {}


def _load_bm25(league: str) -> dict:
    """Load (and cache) the pickled BM25 payload for a league."""
    if league not in _BM25_CACHE:
        path = config.bm25_path(league)
        if not path.exists():
            raise FileNotFoundError(
                f"No BM25 index for {league} at {path}. Run build_index.py."
            )
        with path.open("rb") as f:
            _BM25_CACHE[league] = pickle.load(f)
    return _BM25_CACHE[league]


def dense_candidates(league: str, q_emb, n: int) -> tuple[list[str], dict[str, float]]:
    """Return (ids in descending similarity order, {id: cosine distance})."""
    client = get_client()
    coll = client.get_collection(config.collection_name(league))
    res = coll.query(query_embeddings=[q_emb], n_results=min(n, coll.count()))
    ids = res["ids"][0]
    dists = res["distances"][0]
    return ids, dict(zip(ids, dists))


def expand_bm25_query(query: str) -> str:
    """Append configured synonyms for any matching query term (BM25 side only).

    e.g. "hitting in the PWHL" -> "hitting in the PWHL checking body checking
    contact". The dense embedding sees the original query; only the literal
    keyword search gets the synonym boost. See config.BM25_SYNONYMS.
    """
    tokens = tokenize(query)
    extra: list[str] = []
    for tok in tokens:
        if tok in config.BM25_SYNONYMS:
            extra.extend(config.BM25_SYNONYMS[tok])
    return f"{query} {' '.join(extra)}".strip() if extra else query


def bm25_candidates(league: str, query: str, n: int, zone: str) -> list[str]:
    """Return chunk ids from a BM25 zone ('title' or 'body') by score desc."""
    payload = _load_bm25(league)
    bm25 = payload["bm25_title"] if zone == "title" else payload["bm25_body"]
    scores = bm25.get_scores(tokenize(expand_bm25_query(query)))
    ids = payload["ids"]
    ranked = sorted(range(len(ids)), key=lambda i: scores[i], reverse=True)
    return [ids[i] for i in ranked[:n]]


# ===========================================================================
# Reciprocal Rank Fusion
# ===========================================================================
def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[str]],
    weights: dict[str, float],
    rrf_k: int = config.RRF_K,
) -> tuple[list[str], dict[str, float], dict[str, list[str]]]:
    """Weighted RRF. Each list contributes weight / (rrf_k + rank).

    Returns (fused_ids_in_order, score_by_id, retrievers_by_id).
    """
    scores: dict[str, float] = defaultdict(float)
    retrievers: dict[str, list[str]] = defaultdict(list)
    for method, ids in ranked_lists.items():
        w = weights.get(method, 1.0)
        for rank, _id in enumerate(ids):
            scores[_id] += w / (rrf_k + rank + 1)  # rank is 0-based
            retrievers[_id].append(method)
    fused = sorted(scores, key=lambda i: scores[i], reverse=True)
    return fused, dict(scores), dict(retrievers)


def _fuse_one_league(league: str, q_emb, query: str, n: int):
    """Hybrid candidates for one league, fused. Returns ordered record dicts."""
    payload = _load_bm25(league)
    id2idx = {cid: i for i, cid in enumerate(payload["ids"])}

    dense_ids, dense_dist = dense_candidates(league, q_emb, n)
    title_ids = bm25_candidates(league, query, n, zone="title")
    body_ids = bm25_candidates(league, query, n, zone="body")

    fused, scores, retrievers = reciprocal_rank_fusion(
        {"dense": dense_ids, "bm25_title": title_ids, "bm25_body": body_ids},
        {
            "dense": config.DENSE_WEIGHT,
            "bm25_title": config.BM25_TITLE_WEIGHT,
            "bm25_body": config.BM25_BODY_WEIGHT,
        },
    )

    records = []
    for cid in fused:
        idx = id2idx[cid]
        meta = payload["metas"][idx]
        records.append(
            {
                "league": meta["league"],
                "rule_number": meta["rule_number"],
                "rule_name": meta["rule_name"],
                "section_header": meta["section_header"],
                "text": payload["docs"][idx],
                "score": round(scores[cid], 6),
                "retrievers": retrievers[cid],
                # cosine distance from the dense retriever (None if BM25-only);
                # used by the generation relevance gate.
                "dense_distance": dense_dist.get(cid),
            }
        )
    return records


# ===========================================================================
# Public API
# ===========================================================================
def retrieve(
    question: str,
    leagues: list[str] | None = None,
    k: int = config.TOP_K,
    *,
    prev_question: str | None = None,
    default_leagues: list[str] | None = None,
) -> list[dict]:
    """Hybrid retrieval over one or more leagues.

    If `leagues` is given it is used verbatim (metadata filter). Otherwise the
    target league(s) are detected from the question (and session default).
    Single-league questions return up to `k` chunks. Cross-league questions
    return up to `k` chunks PER detected league, interleaved so each league is
    represented (a comparison needs a healthy set from every side).
    """
    if leagues:
        target = _ordered([lg.upper() for lg in leagues])
        mode = "explicit-arg"
    else:
        target, mode = detect_leagues(question, default_leagues)

    query = expand_query(question, prev_question)
    q_emb = get_embedder().encode(
        [config.BGE_QUERY_INSTRUCTION + query], normalize_embeddings=True
    )[0].tolist()

    per_league = {
        lg: _fuse_one_league(lg, q_emb, query, config.CANDIDATE_K) for lg in target
    }

    # Single league: straight top-k. A genuine multi-league comparison (a
    # handful of explicitly/implied-named leagues): take k PER league (not k
    # total), then interleave so each league is represented up front — the old
    # global k=5 cap starved the second league (e.g. 4 NHL + 1 IIHF).
    # Ambiguous is different: no league was identified, so `target` is ALL
    # leagues as a fallback, not a comparison. Fanning k-per-league out to k*6
    # chunks there just dilutes context and invites citation drift, so keep the
    # global k cap for that case (the system prompt clarifies / defaults NHL).
    if len(target) == 1:
        results = per_league[target[0]][:k]
    elif mode == "ambiguous":
        results = _round_robin(per_league, target, k)
    else:
        capped = {lg: per_league[lg][:k] for lg in target}
        results = _round_robin(capped, target, k * len(target))

    # Tag retrieval metadata on the list via the first element is awkward; we
    # attach detection mode to each record so the UI/eval can surface it.
    for r in results:
        r["detection_mode"] = mode
    return results


def _round_robin(per_league: dict[str, list[dict]], order: list[str], k: int) -> list[dict]:
    """Interleave per-league results so each league contributes before any
    league contributes a second time — guarantees representation up to k."""
    out: list[dict] = []
    idxs = {lg: 0 for lg in order}
    while len(out) < k and any(idxs[lg] < len(per_league[lg]) for lg in order):
        for lg in order:
            if idxs[lg] < len(per_league[lg]):
                out.append(per_league[lg][idxs[lg]])
                idxs[lg] += 1
                if len(out) >= k:
                    break
    return out


# ===========================================================================
# Sanity / demo (Phase 4 acceptance)
# ===========================================================================
def _dense_only(league: str, query: str, k: int) -> list[dict]:
    """Dense-only ranking, for the before/after RRF comparison."""
    q_emb = get_embedder().encode(
        [config.BGE_QUERY_INSTRUCTION + query], normalize_embeddings=True
    )[0].tolist()
    payload = _load_bm25(league)
    id2idx = {cid: i for i, cid in enumerate(payload["ids"])}
    out = []
    dense_ids, _ = dense_candidates(league, q_emb, k)
    for cid in dense_ids:
        m = payload["metas"][id2idx[cid]]
        out.append({"rule_number": m["rule_number"], "rule_name": m["rule_name"]})
    return out


def _rank_of_icing(records: list[dict]) -> str:
    for i, r in enumerate(records):
        if "icing" in r["rule_name"].lower():
            return f"rank {i + 1}"
    return "not in top-k"


def _demo() -> None:
    sep = "=" * 72

    # ---- 1. Fusion fix: before (dense-only) vs after (RRF) for "icing" ------
    print(sep)
    print("1. RRF FUSION FIX — 'icing' on AHL and NCAA (dense-only vs hybrid RRF)")
    print(sep)
    for lg in ["AHL", "NCAA"]:
        dense = _dense_only(lg, "icing", 5)
        rrf = retrieve("icing", leagues=[lg], k=5)
        print(f"\n{lg}:")
        print("  dense-only top-5:")
        for i, r in enumerate(dense):
            print(f"    {i+1}. Rule {r['rule_number']} ({r['rule_name']})")
        print("  hybrid RRF top-5:")
        for i, r in enumerate(rrf):
            tag = "+".join(r["retrievers"])
            print(f"    {i+1}. Rule {r['rule_number']} ({r['rule_name']})  [{tag}]")
        print(f"  >> icing rule: dense {_rank_of_icing(dense)}  ->  RRF {_rank_of_icing(rrf)}")

    # ---- 2. League detection: explicit / implied / ambiguous ----------------
    print("\n" + sep)
    print("2. LEAGUE DETECTION")
    print(sep)
    cases = [
        ("explicit", "What is icing in the NHL?"),
        ("implied", "At what age is body checking allowed in youth hockey?"),
        ("implied", "Is body checking allowed in women's hockey?"),
        ("ambiguous", "What is a two-line pass?"),
    ]
    for label, q in cases:
        leagues, mode = detect_leagues(q)
        shown = leagues if mode != "ambiguous" else f"ALL {len(leagues)} (defer to clarify)"
        print(f"  [{label:9}] {q!r}\n             -> mode={mode}, leagues={shown}")

    # ---- 3. Cross-league: must return BOTH an NHL and a PWHL chunk ----------
    print("\n" + sep)
    print("3. CROSS-LEAGUE — 'Is hitting in the PWHL the same as in the NHL?'")
    print(sep)
    res = retrieve("Is hitting in the PWHL the same as in the NHL?", k=5)
    for r in res:
        print(f"  {r['league']:11} Rule {r['rule_number']} ({r['rule_name']})  score={r['score']}")
    leagues_present = {r["league"] for r in res}
    print(f"  >> leagues present: {sorted(leagues_present)}  "
          f"(NHL & PWHL both present: {{'NHL','PWHL'}}.issubset == "
          f"{ {'NHL','PWHL'}.issubset(leagues_present) })")

    # ---- 4. Return shape (what Phase 5 injects) -----------------------------
    print("\n" + sep)
    print("4. RETURN SHAPE (one record)")
    print(sep)
    import json
    sample = retrieve("icing", leagues=["NHL"], k=1)[0]
    print(json.dumps(sample, indent=2)[:600])


if __name__ == "__main__":
    _demo()
