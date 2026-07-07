"""
QuickWhistle — Phase 8: Evaluation.

Runs the user-provided test set (tests/test_set.jsonl) through the live pipeline
and scores it two ways:

  1. CUSTOM citation-accuracy + behavioral checks (no extra LLM calls):
       - retrieval:  was the EXPECTED rule retrieved (grounded)?
       - citation:   did the model CITE the expected rule?
       - fidelity:   are all cited rules among the retrieved ones? (drift check)
       - behavior:   off-topic -> redirect, empty -> refusal, ambiguous ->
                     ambiguous detection, tournament/format -> decline, etc.

  2. RAGAS (faithfulness, answer relevancy, context recall) on a grounded
     subset, wired to Gemini (rate-limited) + local BGE embeddings.
     context_recall's reference ("ground truth") is built from the EXPECTED
     rule's canonical text in our own corpus (authoritative, not invented);
     only rows with an expected_rule are eligible.

Answers are generated once and cached to tests/eval_run.jsonl so reruns and the
RAGAS pass don't re-spend quota. Calls are spaced for the 5 req/min free tier.

Usage:
  python src/eval.py                       # generate (cached) + custom table/CSV
  python src/eval.py --regenerate          # ignore cache, regenerate answers
  python src/eval.py --ragas               # also run RAGAS on a grounded subset
  python src/eval.py --ragas --ragas-limit 6
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402
from src.answer import answer  # noqa: E402

EVAL_RUN_PATH = config.TESTS_DIR / "eval_run.jsonl"
EVAL_CSV_PATH = config.TESTS_DIR / "eval_results.csv"
SPACING = 13 if config.MODEL_PROVIDER == "gemini" else 0


# ===========================================================================
# Load test set + answer generation (cached)
# ===========================================================================
def load_test_set() -> list[dict]:
    with config.TEST_SET_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def generate_answers(rows: list[dict], regenerate: bool) -> dict[str, dict]:
    """Run each question through answer() once; cache results by id."""
    cache: dict[str, dict] = {}
    if EVAL_RUN_PATH.exists() and not regenerate:
        with EVAL_RUN_PATH.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    cache[r["id"]] = r
    if regenerate and EVAL_RUN_PATH.exists():
        EVAL_RUN_PATH.unlink()

    todo = [r for r in rows if r["id"] not in cache]
    print(f"[eval] {len(cache)} cached, {len(todo)} to generate "
          f"(backend={config.MODEL_PROVIDER})")

    for i, row in enumerate(todo):
        if i and SPACING:
            time.sleep(SPACING)
        # Let the system detect leagues itself (the test set checks detection);
        # do NOT feed expected_leagues.
        try:
            res = answer(row["question"])
        except Exception as e:
            # Most likely the daily free-tier quota. Stop generating, keep what
            # we have, and let scoring proceed on the cached rows. Rerun later
            # (cache resumes) to finish the remainder.
            print(f"  [stop] generation halted at {row['id']}: "
                  f"{type(e).__name__}. {len(cache)} rows cached; rerun later to "
                  f"finish.")
            break
        rec = {
            "id": row["id"],
            "question": row["question"],
            "type": row["type"],
            "expected_leagues": row["expected_leagues"],
            "expected_rule": row["expected_rule"],
            "answer": res["answer"],
            "leagues": res["leagues"],
            "detection_mode": res["detection_mode"],
            "grounded": res["grounded"],
            "chunks": [
                {"league": c["league"], "rule_number": c["rule_number"],
                 "rule_name": c["rule_name"]}
                for c in res["chunks"]
            ],
        }
        cache[row["id"]] = rec
        with EVAL_RUN_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  generated {row['id']} ({i + 1}/{len(todo)})")
    return cache


# ===========================================================================
# Citation parsing
# ===========================================================================
_LEAGUE_KEYS = [
    ("USA_HOCKEY", ["USA HOCKEY", "USAH", "USA "]),
    ("PWHL", ["PWHL"]),
    ("IIHF", ["IIHF"]),
    ("NCAA", ["NCAA"]),
    ("AHL", ["AHL"]),
    ("NHL", ["NHL"]),
]


def _league_in(text: str) -> str | None:
    up = text.upper()
    for code, keys in _LEAGUE_KEYS:
        if any(k in up for k in keys):
            return code
    return None


def parse_cited_rules(answer_text: str) -> list[tuple[str | None, str]]:
    """Extract (league, rule_number) pairs from the answer's Sources block."""
    parts = re.split(r"sources\s*:", answer_text, flags=re.IGNORECASE)
    block = parts[1] if len(parts) > 1 else answer_text
    cites: list[tuple[str | None, str]] = []
    for line in block.splitlines():
        m = re.search(r"rule\s*#?\s*(\d+)", line, flags=re.IGNORECASE)
        if m:
            cites.append((_league_in(line), m.group(1)))
    return cites


# ===========================================================================
# Behavioral checks
# ===========================================================================
_REFUSAL = [
    "wasn't able to find", "was not able to find", "couldn't find",
    "could not find", "consult", "outside the scope", "does not contain",
    "do not have", "not in the retrieved", "unable to",
]
_REDIRECT = [
    "only answer", "can only", "specialized in", "cannot provide",
    "i am an ai assistant", "hockey rules",
]


def _has_any(text: str, phrases: list[str]) -> bool:
    low = text.lower()
    return any(p in low for p in phrases)


# ===========================================================================
# Per-row custom scoring
# ===========================================================================
def score_row(rec: dict) -> dict:
    t = rec["type"]
    answer_text = rec["answer"]
    grounded_rules = {(c["league"], c["rule_number"]) for c in rec["chunks"]}
    used_leagues = set(rec["leagues"])
    cited = parse_cited_rules(answer_text)
    cited_set = {(lg, rn) for lg, rn in cited if lg}

    exp_rule = rec["expected_rule"]
    exp_leagues = rec["expected_leagues"] or []
    exp_pairs = {(lg, exp_rule) for lg in exp_leagues} if exp_rule else set()

    out = {
        "id": rec["id"],
        "type": t,
        "detection_mode": rec["detection_mode"],
        "used_leagues": ",".join(sorted(used_leagues)) or "-",
        "cited": "; ".join(f"{lg or '?'}:{rn}" for lg, rn in cited) or "-",
        # metrics (None = not applicable to this row)
        "retrieval_ok": None,    # expected rule retrieved
        "citation_correct": None,  # model cited the expected rule
        "citation_fidelity": None,  # all cited rules were retrieved
        "behavior_ok": None,
    }

    # Citation fidelity (drift): every cited rule was actually retrieved.
    if cited_set:
        out["citation_fidelity"] = all(c in grounded_rules for c in cited_set)

    if t in ("gameplay", "definitional", "expert_tone", "expert_ref_signal"):
        if exp_pairs:
            out["retrieval_ok"] = bool(exp_pairs & grounded_rules)
            out["citation_correct"] = bool(exp_pairs & cited_set)
        # checking-age (uc3): Junior edition lacks it -> refusal is acceptable
        if rec["id"] == "uc3_checking_age_usa":
            out["behavior_ok"] = _has_any(answer_text, _REFUSAL) or bool(cited_set)
        else:
            out["behavior_ok"] = bool(cited_set) or _has_any(answer_text, _REFUSAL)

    elif t == "cross_league":
        # both expected leagues represented in retrieval
        out["retrieval_ok"] = set(exp_leagues).issubset(used_leagues)
        out["behavior_ok"] = set(exp_leagues).issubset(used_leagues) and bool(cited_set)

    elif t in ("tournament_structure", "refusal_boundary"):
        # must decline to give specifics, not fabricate a format
        out["behavior_ok"] = _has_any(answer_text, _REFUSAL)

    elif t == "ambiguous":
        out["behavior_ok"] = (
            rec["detection_mode"] == "ambiguous" or "?" in answer_text
        )

    elif t == "off_topic":
        # redirect, and no fabricated rule citations
        out["behavior_ok"] = _has_any(answer_text, _REDIRECT) and not cited_set

    elif t == "empty_retrieval":
        out["behavior_ok"] = _has_any(answer_text, _REFUSAL) and not cited_set

    return out


# ===========================================================================
# RAGAS (optional)
# ===========================================================================
def corpus_rule_text(league: str, rule_number: str, limit: int = 1800) -> str:
    """Concatenate a rule's chunk text from our corpus — the recall reference."""
    path = config.CHUNKS_DIR / f"{league}.jsonl"
    if not path.exists():
        return ""
    parts = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            c = json.loads(line)
            if c["rule_number"] == rule_number:
                parts.append(c["text"])
    return " ".join(parts)[:limit]


def run_ragas(cache: dict[str, dict], rows: list[dict], limit: int) -> dict:
    """Run faithfulness, answer_relevancy, context_recall on a grounded subset."""
    from datasets import Dataset
    from langchain_core.embeddings import Embeddings
    from langchain_core.rate_limiters import InMemoryRateLimiter
    from langchain_google_genai import ChatGoogleGenerativeAI
    from ragas import evaluate
    from ragas.embeddings import LangchainEmbeddingsWrapper
    from ragas.llms import LangchainLLMWrapper
    from ragas.metrics import answer_relevancy, context_recall, faithfulness
    from ragas.run_config import RunConfig

    from src.build_index import get_embedder

    # Eligible: grounded answers with an expected_rule (so we can build a
    # corpus-grounded reference for context_recall) and that actually retrieved
    # the expected rule.
    eligible = []
    for r in rows:
        rec = cache.get(r["id"])
        if not rec or not rec["expected_rule"] or not rec["grounded"]:
            continue
        gr = {(c["league"], c["rule_number"]) for c in rec["chunks"]}
        exp = {(lg, rec["expected_rule"]) for lg in (rec["expected_leagues"] or [])}
        if exp & gr:
            eligible.append(rec)
    subset = eligible[:limit]
    if not subset:
        print("[ragas] no eligible grounded rows; skipping.")
        return {}

    print(f"[ragas] evaluating {len(subset)} rows: "
          f"{[r['id'] for r in subset]}")

    questions, answers, contexts, refs = [], [], [], []
    for rec in subset:
        # rebuild the retrieved context texts from a fresh retrieval (cheap,
        # local) so RAGAS sees the same chunks the model was given.
        from src.retrieve import retrieve
        ch = retrieve(rec["question"], k=config.TOP_K)
        questions.append(rec["question"])
        answers.append(rec["answer"])
        contexts.append([c["text"] for c in ch])
        lg = (rec["expected_leagues"] or [None])[0]
        refs.append(corpus_rule_text(lg, rec["expected_rule"]))

    ds = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": refs,
    })

    # langchain-google-genai 1.0.x forwards call-time sampling kwargs (RAGAS
    # passes `temperature`) straight to the low-level client.generate_content(),
    # which rejects them -> TypeError -> NaN scores. Fold them into
    # generation_config where they belong.
    class _PatchedGemini(ChatGoogleGenerativeAI):
        _SAMPLING = ("temperature", "top_p", "top_k", "max_output_tokens")

        def _generate(self, messages, stop=None, run_manager=None, *,
                      generation_config=None, **kwargs):
            gc = dict(generation_config or {})
            if "max_tokens" in kwargs:
                gc.setdefault("max_output_tokens", kwargs.pop("max_tokens"))
            for k in self._SAMPLING:
                if k in kwargs:
                    gc.setdefault(k, kwargs.pop(k))
            return super()._generate(
                messages, stop=stop, run_manager=run_manager,
                generation_config=gc, **kwargs,
            )

    # Throttle every Gemini call RAGAS makes to stay under 5 req/min.
    limiter = InMemoryRateLimiter(
        requests_per_second=0.08, check_every_n_seconds=0.5, max_bucket_size=1
    )
    llm = _PatchedGemini(
        model=config.GEMINI_MODEL,
        google_api_key=config.GEMINI_API_KEY,
        temperature=0.0,
        max_output_tokens=2048,
        rate_limiter=limiter,
    )

    class BGEEmbeddings(Embeddings):
        def __init__(self):
            self.m = get_embedder()

        def embed_documents(self, texts):
            return self.m.encode(texts, normalize_embeddings=True).tolist()

        def embed_query(self, text):
            v = self.m.encode(
                [config.BGE_QUERY_INSTRUCTION + text], normalize_embeddings=True
            )
            return v[0].tolist()

    ragas_llm = LangchainLLMWrapper(llm)
    ragas_emb = LangchainEmbeddingsWrapper(BGEEmbeddings())

    result = evaluate(
        ds,
        metrics=[faithfulness, answer_relevancy, context_recall],
        llm=ragas_llm,
        embeddings=ragas_emb,
        run_config=RunConfig(max_workers=1, timeout=600),
        raise_exceptions=False,
    )
    df = result.to_pandas()
    return {"ids": [r["id"] for r in subset], "df": df}


# ===========================================================================
# Reporting
# ===========================================================================
def _fmt(v) -> str:
    if v is None:
        return " · "
    if isinstance(v, bool):
        return " ✓ " if v else " ✗ "
    return str(v)


def print_table(scored: list[dict]) -> None:
    print("\n" + "=" * 100)
    print("CUSTOM CITATION-ACCURACY + BEHAVIORAL CHECKS (per question)")
    print("=" * 100)
    hdr = f"{'id':28} {'type':18} {'detect':12} {'retr':5} {'cite=exp':9} {'fidel':6} {'behav':6}"
    print(hdr)
    print("-" * 100)
    for s in scored:
        print(f"{s['id']:28} {s['type']:18} {s['detection_mode']:12} "
              f"{_fmt(s['retrieval_ok']):5} {_fmt(s['citation_correct']):9} "
              f"{_fmt(s['citation_fidelity']):6} {_fmt(s['behavior_ok']):6}")


def _rate(scored, key) -> str:
    vals = [s[key] for s in scored if s[key] is not None]
    if not vals:
        return "n/a"
    return f"{sum(vals)}/{len(vals)} ({100*sum(vals)//len(vals)}%)"


def print_aggregates(scored: list[dict]) -> None:
    print("\n" + "=" * 60)
    print("AGGREGATE SCORES (custom)")
    print("=" * 60)
    print(f"  Retrieval accuracy (expected rule retrieved): {_rate(scored,'retrieval_ok')}")
    print(f"  Citation correctness (cited == expected):     {_rate(scored,'citation_correct')}")
    print(f"  Citation fidelity (cited ⊆ retrieved):        {_rate(scored,'citation_fidelity')}")
    print(f"  Behavioral correctness (all types):           {_rate(scored,'behavior_ok')}")


def write_csv(scored: list[dict]) -> None:
    with EVAL_CSV_PATH.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(scored[0].keys()))
        w.writeheader()
        w.writerows(scored)
    print(f"\n[eval] wrote per-question CSV -> {EVAL_CSV_PATH}")


def main() -> None:
    ap = argparse.ArgumentParser(description="QuickWhistle evaluation")
    ap.add_argument("--regenerate", action="store_true", help="Ignore cached answers")
    ap.add_argument("--ragas", action="store_true", help="Also run RAGAS (throttled)")
    ap.add_argument("--ragas-limit", type=int, default=4, help="RAGAS subset size")
    args = ap.parse_args()

    rows = load_test_set()
    cache = generate_answers(rows, args.regenerate)
    done = [r for r in rows if r["id"] in cache]
    missing = [r["id"] for r in rows if r["id"] not in cache]
    scored = [score_row(cache[r["id"]]) for r in done]
    if missing:
        print(f"\n[eval] NOTE: {len(missing)} rows not yet generated "
              f"(quota): {missing}\n       Rerun `python src/eval.py` later to "
              f"finish; cached rows are reused.")

    print_table(scored)
    print_aggregates(scored)
    write_csv(scored)

    if args.ragas:
        res = run_ragas(cache, rows, args.ragas_limit)
        if res:
            df = res["df"]
            df.insert(0, "id", res["ids"])
            cols = [c for c in ["id", "faithfulness", "answer_relevancy",
                                "context_recall"] if c in df.columns]
            ragas_csv = config.TESTS_DIR / "ragas_results.csv"
            df[cols].to_csv(ragas_csv, index=False)  # persist (flush-safe)
            print("\n" + "=" * 60)
            print("RAGAS (per question)")
            print("=" * 60)
            print(df[cols].to_string(index=False), flush=True)
            print("\nRAGAS means:", flush=True)
            for c in cols[1:]:
                print(f"  {c}: {df[c].mean():.3f}", flush=True)
            print(f"\n[eval] wrote RAGAS CSV -> {ragas_csv}", flush=True)


if __name__ == "__main__":
    main()
