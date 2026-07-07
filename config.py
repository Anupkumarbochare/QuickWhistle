"""
QuickWhistle central configuration.

Everything that another module might want to tune lives here so the rest of the
codebase imports from one place. Secrets are NOT stored here — they come from
the environment (.env), loaded below with python-dotenv.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the project root (the folder this file lives in).
PROJECT_ROOT = Path(__file__).resolve().parent
load_dotenv(PROJECT_ROOT / ".env")


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"          # source rulebook PDFs (gitignored, copyrighted)
CHUNKS_DIR = DATA_DIR / "chunks"    # {league}.jsonl output of ingestion
CHROMA_DIR = DATA_DIR / "chroma"    # persistent Chroma vector store (gitignored)

PROMPTS_DIR = PROJECT_ROOT / "prompts"
SYSTEM_PROMPT_PATH = PROMPTS_DIR / "system_prompt.txt"

TESTS_DIR = PROJECT_ROOT / "tests"
TEST_SET_PATH = TESTS_DIR / "test_set.jsonl"

# Opt-in long-term memory store (one JSON file per user). Off by default.
USER_MEMORY_DIR = DATA_DIR / "user_memory"


# ---------------------------------------------------------------------------
# Leagues — one Chroma collection per league
# ---------------------------------------------------------------------------
LEAGUES = ["NHL", "PWHL", "IIHF", "AHL", "NCAA", "USA_HOCKEY"]

# Map a league code to the raw PDF filename in data/raw/. Adjust if your files
# are named differently — only this dict needs to change.
RAW_PDF_FILES = {
    "NHL": "NHL rule book.pdf",
    "PWHL": "2025-2026_PWHL_Rulebook.pdf",
    "IIHF": "2025-26_iihf_rulebook_22122025-v1.pdf",
    "AHL": "2025-26_AHLRuleBook.pdf",
    "NCAA": "NCAA_rule_book.pdf",
    "USA_HOCKEY": "2025-29_USAH_Playing_Rules_-_Junior.pdf",
}

# Human-readable rulebook name used in citations (Sources block).
LEAGUE_DISPLAY = {
    "NHL": "NHL Official Rules 2025-2026",
    "PWHL": "PWHL Rulebook 2025-2026",
    "IIHF": "IIHF Official Rule Book 2025-26",
    "AHL": "AHL Rulebook 2025-26",
    "NCAA": "NCAA Ice Hockey Rules",
    "USA_HOCKEY": "USA Hockey Playing Rules 2025-29",
}


# ---------------------------------------------------------------------------
# Ingestion / chunking
# ---------------------------------------------------------------------------
# If a page yields fewer than this many characters from PyMuPDF, we treat it as
# possibly image-only and fall back (pdfplumber, then OCR).
MIN_PAGE_CHARS = 15
# A single rule longer than this many characters is recursively sub-split so it
# fits the embedding context. Every sub-chunk keeps the same rule metadata.
MAX_CHUNK_CHARS = 1800
# Character overlap between sub-chunks so context isn't lost at the seam.
SUBCHUNK_OVERLAP = 150


# ---------------------------------------------------------------------------
# Embeddings (local, free)
# ---------------------------------------------------------------------------
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384  # bge-small-en-v1.5 output dimension
# BGE retrieval models expect this instruction prefixed to the QUERY only
# (passages/documents are embedded without any prefix).
BGE_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


def collection_name(league: str) -> str:
    """Chroma collection name for a league (one collection per league)."""
    return f"rules_{league.lower()}"


def bm25_path(league: str):
    """Path to the persisted BM25 index for a league."""
    return CHROMA_DIR / f"bm25_{league.lower()}.pkl"


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K = 5                 # final chunks returned to the LLM
RRF_K = 60                # reciprocal rank fusion constant (standard default)
CANDIDATE_K = 20          # candidates pulled from each retriever before fusion

# Weighted reciprocal rank fusion: each ranked list contributes
#   weight / (RRF_K + rank).
# The keyword (BM25) signal is split into two zones so a rule-TITLE match
# ("icing" -> the rule literally named Icing) outranks a body chunk that merely
# mentions the word (e.g. an officials' "Signals" rule that lists icing among
# hand signals). All three lists feed one RRF; still BM25 + dense, no new
# retriever types. Equal weights by default.
DENSE_WEIGHT = 1.0
BM25_TITLE_WEIGHT = 1.0   # BM25 over rule_name + section_header
BM25_BODY_WEIGHT = 1.0    # BM25 over chunk body text

# Query-expansion synonyms applied to the BM25 (keyword) side ONLY. The dense
# embedding already captures semantics, but BM25 is literal — so casual phrasing
# ("hitting") won't match the formal rule term ("body checking") without help.
# Each key term, when present in the query, appends its extra terms to the
# keyword query. Multi-word values are fine (they get tokenized). Extend freely.
BM25_SYNONYMS = {
    "hitting": ["checking", "body checking", "contact"],
    "hit": ["check", "body check"],
    "fighting": ["fisticuffs"],
    "fight": ["fisticuffs", "altercation"],
}

# Relevance gate (safety net only). With bge-small, dense distances cluster
# ~0.30-0.52 whether or not a chunk is truly on-topic, so distance alone cannot
# cleanly separate "off-topic" from "no match". The PRIMARY refusal/redirect
# behavior is therefore the LLM's, driven by the verbatim system prompt. This
# gate fires only on degenerate, very-far retrieval as a backstop.
RELEVANCE_MAX_DISTANCE = 0.70


# ---------------------------------------------------------------------------
# Generation LLM — swappable via a single config value.
# Set MODEL_PROVIDER in .env to "gemini" (default) or "ollama".
# ---------------------------------------------------------------------------
MODEL_PROVIDER = os.getenv("MODEL_PROVIDER", "gemini").lower()

# The actual model name for the chosen provider.
# gemini-1.5-flash (the brief's stated default) was retired by Google. This
# key's free tier is capped at ~20 requests/day for gemini-2.5-flash, so the
# default is gemini-2.5-flash-lite (separate daily bucket). Override via .env.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

# Convenience: the single "MODEL" value the brief refers to.
MODEL = GEMINI_MODEL if MODEL_PROVIDER == "gemini" else OLLAMA_MODEL

# Secret — never hardcoded. Only needed when MODEL_PROVIDER == "gemini".
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Generation parameters
TEMPERATURE = 0.1         # low: we want grounded, deterministic answers
# Generous budget: gemini-2.5-flash is a "thinking" model whose internal
# reasoning tokens also draw from this budget, so a small cap truncates the
# visible answer. The system prompt still constrains answers to ~200 words.
MAX_OUTPUT_TOKENS = 4096
