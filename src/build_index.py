"""
QuickWhistle — Phase 3: Index building.

data/chunks/{league}.jsonl
   -> embed each chunk with BGE (BAAI/bge-small-en-v1.5)
   -> one persistent Chroma collection per league   (dense / semantic search)
   -> one persisted BM25 index per league           (keyword search)

Both halves feed the hybrid retriever in Phase 4. Everything is local and free:
embeddings run on-device, Chroma persists to data/chroma/, and the BM25 index is
pickled alongside it.

The build is idempotent: if a league's collection already holds the expected
number of chunks (and the BM25 file exists), it is skipped unless you pass
--force. This is what lets the store reload without re-embedding.

Usage:
  python src/build_index.py --all                 # build every league
  python src/build_index.py --league NHL          # one league
  python src/build_index.py --all --force         # rebuild from scratch
  python src/build_index.py --sanity              # quick "icing" retrieval check
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path

# Make `config` importable whether run as a script or a module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


# ===========================================================================
# Loading chunks
# ===========================================================================
def load_chunks(league: str) -> list[dict]:
    """Read data/chunks/{league}.jsonl into a list of chunk dicts."""
    path = config.CHUNKS_DIR / f"{league}.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"No chunks for {league} at {path}. Run ingest.py first."
        )
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


# ===========================================================================
# BM25 (keyword) index
# ===========================================================================
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_HYPHENATED_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)+")


def tokenize(text: str) -> list[str]:
    """Lowercase tokenization shared by indexing and retrieval.

    Hyphenated rule terms ("Off-side", "Cross-checking") also emit a joined form
    ("offside", "crosschecking") so a user typing the unhyphenated word still
    matches the rule. Without this, "offside" (one token) never matches the
    rulebook's "off"+"side", and the right rule gets out-ranked.
    """
    low = text.lower()
    tokens = _TOKEN_RE.findall(low)
    tokens += [m.group(0).replace("-", "") for m in _HYPHENATED_RE.finditer(low)]
    return tokens


def build_bm25(league: str, chunks: list[dict]) -> None:
    """Build and persist zone-weighted BM25 indexes plus parallel chunk records.

    Two BM25 zones are built so the retriever can weight a rule-TITLE match
    above a mere body mention:
      * bm25_title - tokens from rule_name + section_header
      * bm25_body  - tokens from the chunk body text
    Both, plus ids/documents/metadatas in document order, are pickled together
    so the retriever can map a hit back to its chunk.
    """
    from rank_bm25 import BM25Okapi

    ids = [f"{league}-{i}" for i in range(len(chunks))]
    docs = [c["text"] for c in chunks]
    metas = [
        {
            "league": c["league"],
            "rule_number": c["rule_number"],
            "rule_name": c["rule_name"],
            "section_header": c["section_header"],
        }
        for c in chunks
    ]
    title_tokens = [
        tokenize(f"{c['rule_name']} {c['section_header']}") for c in chunks
    ]
    body_tokens = [tokenize(c["text"]) for c in chunks]
    bm25_title = BM25Okapi(title_tokens)
    bm25_body = BM25Okapi(body_tokens)

    payload = {
        "bm25_title": bm25_title,
        "bm25_body": bm25_body,
        "ids": ids,
        "docs": docs,
        "metas": metas,
    }
    out = config.bm25_path(league)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        pickle.dump(payload, f)
    print(f"[index] {league}: BM25 zones (title+body, {len(ids)} docs) -> {out.name}")


# ===========================================================================
# Dense (Chroma) index
# ===========================================================================
_EMBEDDER = None


def get_embedder():
    """Lazily load the BGE model once and reuse it across leagues."""
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer

        print(f"[index] loading embedding model {config.EMBED_MODEL} ...")
        _EMBEDDER = SentenceTransformer(config.EMBED_MODEL)
    return _EMBEDDER


def embed_documents(texts: list[str]):
    """Embed passages (no BGE instruction prefix — that's query-only)."""
    model = get_embedder()
    return model.encode(
        texts,
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,  # cosine similarity via normalized vectors
    ).tolist()


def get_client():
    """Persistent Chroma client rooted at data/chroma/."""
    import chromadb
    from chromadb.config import Settings

    config.CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    # Disable anonymized telemetry (avoids noisy posthog warnings; keeps it local).
    return chromadb.PersistentClient(
        path=str(config.CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )


def build_dense(league: str, chunks: list[dict], client, force: bool) -> bool:
    """Build the per-league Chroma collection. Returns True if it (re)embedded."""
    name = config.collection_name(league)
    existing = {c.name for c in client.list_collections()}

    if name in existing and not force:
        coll = client.get_collection(name)
        if coll.count() == len(chunks):
            print(
                f"[index] {league}: dense collection '{name}' already has "
                f"{coll.count()} chunks — skipping embed (use --force to rebuild)."
            )
            return False
        # Count mismatch: rebuild to stay consistent with the chunk file.
        print(f"[index] {league}: count mismatch, rebuilding '{name}'.")
        client.delete_collection(name)
    elif name in existing and force:
        client.delete_collection(name)

    coll = client.create_collection(
        name=name, metadata={"hnsw:space": "cosine", "league": league}
    )

    ids = [f"{league}-{i}" for i in range(len(chunks))]
    docs = [c["text"] for c in chunks]
    metas = [
        {
            "league": c["league"],
            "rule_number": c["rule_number"],
            "rule_name": c["rule_name"],
            "section_header": c["section_header"],
        }
        for c in chunks
    ]
    print(f"[index] {league}: embedding {len(docs)} chunks with BGE ...")
    embeddings = embed_documents(docs)
    # Add in batches to keep memory modest.
    B = 256
    for i in range(0, len(ids), B):
        coll.add(
            ids=ids[i : i + B],
            documents=docs[i : i + B],
            embeddings=embeddings[i : i + B],
            metadatas=metas[i : i + B],
        )
    print(f"[index] {league}: dense collection '{name}' now has {coll.count()} chunks.")
    return True


# ===========================================================================
# Driver
# ===========================================================================
def build_league(league: str, client, force: bool) -> None:
    chunks = load_chunks(league)
    rebuilt = build_dense(league, chunks, client, force)
    # (Re)build BM25 if we rebuilt dense, or if the BM25 file is missing.
    if rebuilt or force or not config.bm25_path(league).exists():
        build_bm25(league, chunks)
    else:
        print(f"[index] {league}: BM25 index present — skipping.")


def sanity_check(client, query: str = "icing") -> None:
    """Embed `query` once and show the top hit per league (no corpus re-embed)."""
    from sentence_transformers import SentenceTransformer  # noqa: F401

    model = get_embedder()
    q_emb = model.encode(
        [config.BGE_QUERY_INSTRUCTION + query], normalize_embeddings=True
    ).tolist()

    print("\n" + "=" * 70)
    print(f"SANITY CHECK — dense top hit for query: {query!r}")
    print("=" * 70)
    for league in config.LEAGUES:
        name = config.collection_name(league)
        try:
            coll = client.get_collection(name)
        except Exception:
            print(f"{league:11} (no collection — build it first)")
            continue
        res = coll.query(query_embeddings=q_emb, n_results=1)
        meta = res["metadatas"][0][0]
        dist = res["distances"][0][0]
        print(
            f"{league:11} -> Rule {meta['rule_number']} "
            f"({meta['rule_name']})  [cosine dist {dist:.3f}]"
        )


def main() -> None:
    ap = argparse.ArgumentParser(description="QuickWhistle index builder")
    ap.add_argument("--league", help="League code, e.g. NHL")
    ap.add_argument("--all", action="store_true", help="Build all leagues")
    ap.add_argument("--force", action="store_true", help="Rebuild even if present")
    ap.add_argument(
        "--sanity", action="store_true", help="Run an 'icing' retrieval sanity check"
    )
    args = ap.parse_args()

    client = get_client()

    if args.sanity and not (args.all or args.league):
        sanity_check(client)
        return

    if args.all:
        targets = list(config.LEAGUES)
    elif args.league:
        targets = [args.league.upper()]
    else:
        ap.error("Provide --league NHL, --all, or --sanity")

    for league in targets:
        build_league(league, client, args.force)

    if args.sanity:
        sanity_check(client)


if __name__ == "__main__":
    main()
