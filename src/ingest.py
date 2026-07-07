"""
QuickWhistle — Phase 2: Ingestion.

PDF  ->  clean text  ->  structure-aware chunks  ->  data/chunks/{league}.jsonl

Strategy (per the build brief, Section 5):
  * STRUCTURE-AWARE primary: split on rule-number / section-header boundaries so
    that one chunk == one rule, carrying citable metadata.
  * RECURSIVE fallback: if a single rule is longer than the embedding context,
    sub-split it but stamp the SAME metadata on every sub-chunk so it stays
    citable.

Each output line is one JSON object:
  {"league", "rule_number", "rule_name", "section_header", "text"}

Text extraction order per page:
  1. PyMuPDF (fitz)            — fast, handles the vast majority of pages.
  2. pdfplumber               — better on table-like pages.
  3. pytesseract (OCR)        — LAST resort, only when a page has no text layer.

Usage:
  python src/ingest.py --league NHL          # one league
  python src/ingest.py --league NHL --sample # print sample chunks, don't write
  python src/ingest.py --all                 # every configured league
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable, Iterable

import fitz  # PyMuPDF

# Make `config` importable whether run as `python src/ingest.py` or as a module.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402


# ===========================================================================
# Data model
# ===========================================================================
@dataclass
class Chunk:
    league: str
    rule_number: str
    rule_name: str
    section_header: str
    text: str


# ===========================================================================
# Per-league parsing rules
# ===========================================================================
# Each league's rulebook is laid out differently, so the header regexes and the
# repeated page-noise lines live in a small per-league spec. NHL is implemented
# and tuned now (brief: "Start with the NHL book only"). The other five are
# registered as TODO so the dispatcher fails loudly instead of silently
# mis-parsing — they get tuned in a later pass after NHL is reviewed.
@dataclass
class LeagueSpec:
    # Matches a section banner in the body, e.g. "SECTION 5 - OFFICIALS".
    # group(1)=number (may be a spelled-out word for USA Hockey).
    # Optional named group 'sname' captures the section name when it's inline;
    # otherwise the chunker reads the name from the next content line.
    section_re: re.Pattern
    # Matches a rule header, e.g. "Rule 30 – Appointment of Officials" or a bare
    # "Rule 30" / "RULE 30". group(1)=number. Optional named group 'name'
    # captures an inline rule name; otherwise the name is read from the next
    # content line. This single mechanism handles both layout families:
    #   - inline name   (NHL, AHL, NCAA): name is in the header line
    #   - name-next-line (PWHL, IIHF, USA Hockey): header line is just the number
    rule_re: re.Pattern
    # Lines matching any of these (after strip) are dropped as page noise
    # (running headers, footers, copyright banners).
    noise_res: list[re.Pattern]
    # Substrings/patterns scrubbed *inline* anywhere they appear (e.g. clickable
    # navigation link text that gets extracted mid-sentence).
    inline_noise_res: list[re.Pattern]


def _ci(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


# Noise common to most books: bare page numbers and blank lines.
_COMMON_NOISE = [
    re.compile(r"^\s*\d{1,4}\s*$"),  # bare page numbers
    re.compile(r"^\s*$"),            # blank lines
]

# Dash class reused across specs: hyphen-minus, en dash, em dash.
_DASH = r"[\-–—]"


# --- NHL: "SECTION 5 - OFFICIALS" / "Rule 30 – Appointment of Officials" -----
NHL_SPEC = LeagueSpec(
    section_re=_ci(rf"^SECTION\s+(\d+)\s*{_DASH}\s*(?P<sname>.+?)\s*$"),
    rule_re=_ci(rf"^Rule\s+(\d+)\s*{_DASH}\s*(?P<name>.+?)\s*$"),
    noise_res=[
        _ci(r"^NATIONAL HOCKEY LEAGUE\s*$"),
        _ci(r"^OFFICIAL RULES\s+\d{4}-\d{4}\s*$"),
        *_COMMON_NOISE,
    ],
    inline_noise_res=[
        _ci(r"\bPrevious Page\b"),
        _ci(r"\bNext Page\b"),
        _ci(r"\bTable of Contents\b"),
    ],
)

# --- AHL: same family as NHL ("SECTION 1 – PLAYING AREA" / "Rule 1 – Rink") --
AHL_SPEC = LeagueSpec(
    section_re=_ci(rf"^SECTION\s+(\d+)\s*{_DASH}\s*(?P<sname>.+?)\s*$"),
    rule_re=_ci(rf"^Rule\s+(\d+)\s*{_DASH}\s*(?P<name>.+?)\s*$"),
    noise_res=[
        _ci(r"^American Hockey League\s*$"),
        _ci(r"^Official Rules\s+\d{4}-\d{2,4}\s*$"),
        *_COMMON_NOISE,
    ],
    inline_noise_res=[],
)

# --- NCAA: "SECTION 1" or "SECTION 1 / Playing Area"; "RULE 1 - RINK" --------
NCAA_SPEC = LeagueSpec(
    section_re=_ci(r"^SECTION\s+(\d+)\s*(?:/\s*(?P<sname>.+?))?\s*$"),
    rule_re=_ci(rf"^RULE\s+(\d+)\s*{_DASH}\s*(?P<name>.+?)\s*$"),
    noise_res=[
        _ci(r"^\d{4}-\d{2,4}\s+NCAA\b.*$"),
        *_COMMON_NOISE,
    ],
    inline_noise_res=[],
)

# --- PWHL: "SECTION 1: PLAYING AREA"; "Rule 1" (name on next line) -----------
PWHL_SPEC = LeagueSpec(
    section_re=_ci(r"^SECTION\s+(\d+)\s*:\s*(?P<sname>.+?)\s*$"),
    rule_re=_ci(rf"^Rule\s+(\d+)\s*(?:{_DASH}\s*(?P<name>.+?))?\s*$"),
    noise_res=[
        _ci(r"^PWHL Official Rule Book\b.*$"),
        *_COMMON_NOISE,
    ],
    inline_noise_res=[],
)

# --- IIHF: "SECTION 01. PLAYING AREA"; "RULE 14" (name on next line) ---------
# Section numbers are zero-padded ("01"); we normalize them to "1" in the
# chunker. The name is optional here because bare "SECTION 01" divider lines
# exist — those fall back to reading the name from the next content line.
IIHF_SPEC = LeagueSpec(
    section_re=_ci(r"^SECTION\s+(\d{1,2})\.?\s*(?P<sname>.+)?$"),
    rule_re=_ci(r"^RULE\s+(\d+)\s*$"),
    noise_res=[
        _ci(r"^IIHF OFFICIAL RULE BOOK\b.*$"),
        _ci(r"^TABLE OF CONTENTS\s*$"),
        *_COMMON_NOISE,
    ],
    inline_noise_res=[],
)

# --- USA Hockey: "SECTION ONE" (spelled out); "Rule 101." (name inline OR
#     on next line); body uses (a)/(b) sub-lettering, 3-digit rule numbers. ---
USAH_SPEC = LeagueSpec(
    section_re=_ci(r"^SECTION\s+([A-Z]+)\s*$"),
    rule_re=_ci(r"^Rule\s+(\d+)\.\s*(?P<name>.*?)\s*$"),
    noise_res=[
        _ci(r"^SECTION\s+[A-Z]+\s*[—–-].*$"),     # running page header w/ name
        _ci(r"^\d{4}-\d{2}\s+Official Rules\b.*$"),
        *_COMMON_NOISE,
    ],
    inline_noise_res=[],
)

LEAGUE_SPECS: dict[str, LeagueSpec] = {
    "NHL": NHL_SPEC,
    "AHL": AHL_SPEC,
    "NCAA": NCAA_SPEC,
    "PWHL": PWHL_SPEC,
    "IIHF": IIHF_SPEC,
    "USA_HOCKEY": USAH_SPEC,
}


# ===========================================================================
# Text extraction (with graceful fallbacks)
# ===========================================================================
def _ocr_page(page: "fitz.Page") -> str:
    """OCR a single page. Imported lazily so the dep is optional at runtime."""
    try:
        import io

        import pytesseract
        from PIL import Image
    except ImportError:
        return ""  # OCR not available; caller already has empty text
    pix = page.get_pixmap(dpi=200)
    img = Image.open(io.BytesIO(pix.tobytes("png")))
    return pytesseract.image_to_string(img)


def _pdfplumber_page(pdf_path: Path, page_index: int) -> str:
    """Extract one page with pdfplumber (better on table-like layouts)."""
    try:
        import pdfplumber
    except ImportError:
        return ""
    with pdfplumber.open(str(pdf_path)) as pdf:
        if page_index < len(pdf.pages):
            return pdf.pages[page_index].extract_text() or ""
    return ""


def extract_pages(pdf_path: Path) -> list[str]:
    """Return cleaned-ish text for every page, using fallbacks where needed."""
    doc = fitz.open(str(pdf_path))
    pages: list[str] = []
    for i in range(doc.page_count):
        page = doc[i]
        text = page.get_text()
        if len(text.strip()) < config.MIN_PAGE_CHARS:
            # Page may be a table image or scanned — try pdfplumber, then OCR.
            alt = _pdfplumber_page(pdf_path, i)
            if len(alt.strip()) >= config.MIN_PAGE_CHARS:
                text = alt
            else:
                ocr = _ocr_page(page)
                if len(ocr.strip()) > len(text.strip()):
                    text = ocr
        pages.append(text)
    doc.close()
    return pages


# ===========================================================================
# Cleaning + chunking
# ===========================================================================
def _is_toc_line(line: str) -> bool:
    """Table-of-contents lines use dotted leaders, e.g. 'Rule 1 – Rink ..... 1'."""
    return "...." in line or "…" in line


# Control chars (backspace \x08 etc.) and exotic unicode spaces show up in some
# PDFs (notably NCAA). Normalize them before pattern-matching.
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
_USPACE_RE = re.compile(r"[  -   　\t]")


def _normalize(line: str) -> str:
    line = _CTRL_RE.sub("", line)
    line = _USPACE_RE.sub(" ", line)
    return line.strip()


def clean_lines(pages: Iterable[str], spec: LeagueSpec) -> list[str]:
    """Flatten pages to lines, dropping page noise and TOC entries."""
    out: list[str] = []
    for page in pages:
        for raw in page.splitlines():
            line = _normalize(raw)
            if not line:
                continue
            if _is_toc_line(line):
                continue
            if any(p.match(line) for p in spec.noise_res):
                continue
            # Scrub inline navigation/link noise, then collapse the gap it left.
            for p in spec.inline_noise_res:
                line = p.sub(" ", line)
            line = re.sub(r"\s{2,}", " ", line).strip()
            if not line:
                continue
            out.append(line)
    return out


def _recursive_split(text: str, max_chars: int, overlap: int) -> list[str]:
    """Sub-split overly long rule text on paragraph/sentence-ish boundaries.

    Greedy: accumulate paragraphs until the budget is hit, then start a new
    window carrying `overlap` characters of tail context.
    """
    if len(text) <= max_chars:
        return [text]

    # Prefer paragraph boundaries; fall back to sentence-ish, then hard slice.
    paras = re.split(r"\n\s*\n|(?<=\.)\s{2,}", text)
    windows: list[str] = []
    cur = ""
    for para in paras:
        para = para.strip()
        if not para:
            continue
        if len(cur) + len(para) + 1 <= max_chars:
            cur = f"{cur}\n{para}".strip()
        else:
            if cur:
                windows.append(cur)
                cur = (cur[-overlap:] + "\n" + para).strip() if overlap else para
            else:
                # Single paragraph longer than the budget: hard-slice it.
                for j in range(0, len(para), max_chars - overlap):
                    windows.append(para[j : j + max_chars])
                cur = ""
    if cur:
        windows.append(cur)
    return windows or [text]


def _is_header(line: str, spec: LeagueSpec) -> bool:
    """True if the line is itself a section or rule header (used for lookahead)."""
    return bool(spec.section_re.match(line) or spec.rule_re.match(line))


def _optional_group(match: "re.Match", name: str) -> str:
    """Safely fetch an optional named group, returning '' if absent/empty."""
    if name not in match.re.groupindex:
        return ""
    return (match.group(name) or "").strip()


def _looks_like_name(s: str) -> bool:
    """Heuristic: is `s` a plausible rule/section *name* vs. a body sentence?

    Rule names are short title-ish phrases ("Goal Posts and Nets"). Body lines
    wrap long, start lowercase, or contain sentence punctuation. This guards the
    name-on-next-line lookahead from swallowing a paragraph when a cross-
    reference like "Rule 83" accidentally starts a line.
    """
    if not s or len(s) > 60:
        return False
    if "." in s:                       # body sentences carry periods; names don't
        return False
    if s[:1].islower():                # wrapped body lines often start lowercase
        return False
    return True


def _norm_section_no(num: str) -> str:
    """Strip leading zeros from numeric section labels ('01' -> '1')."""
    return str(int(num)) if num.isdigit() else num


# Acronyms kept uppercase when title-casing; small words kept lowercase
# (except as the first word).
_TITLE_ACRONYMS = {"USA", "NHL", "PWHL", "IIHF", "AHL", "NCAA", "TV", "OT"}
_TITLE_SMALL = {
    "a", "an", "and", "or", "the", "of", "to", "in", "on",
    "for", "from", "with", "by", "at", "as",
}


def _cap_token(tok: str) -> str:
    """Capitalize a single alphabetic token, preserving known acronyms."""
    if not tok:
        return tok
    if tok.upper() in _TITLE_ACRONYMS:
        return tok.upper()
    return tok[:1].upper() + tok[1:].lower()


def title_case_name(name: str) -> str:
    """Title-case an ALL-CAPS rule name for the user-facing Sources line.

    Only normalizes names that are entirely uppercase (NCAA/IIHF). Mixed-case
    names from other books are left untouched. Hyphen/slash compounds keep each
    part capitalized ("CROSS-CHECKING" -> "Cross-Checking"), acronyms like USA
    stay uppercase, and minor words stay lowercase unless they lead.
    """
    if not name or not name.isupper():
        return name
    words = name.split()
    out: list[str] = []
    for idx, w in enumerate(words):
        # Split on hyphens/slashes but keep the separators in place.
        parts = re.split(r"([/-])", w)
        rebuilt = "".join(p if p in "/-" else _cap_token(p) for p in parts)
        # Demote minor words (only when not the first word and not a compound).
        if idx != 0 and len(parts) == 1 and w.lower() in _TITLE_SMALL:
            rebuilt = w.lower()
        out.append(rebuilt)
    return " ".join(out)


def chunk_lines(lines: list[str], league: str, spec: LeagueSpec) -> list[Chunk]:
    """Structure-aware chunking: one chunk per rule, with recursive fallback.

    Handles both header layouts. When a section/rule header carries its name
    inline, we use it; when the header line is just a number, we read the name
    from the next content line (unless that line is itself a header).
    """
    chunks: list[Chunk] = []
    cur_section = ""        # most recent "Section N - NAME"
    cur_section_no = ""     # tracked so a bare re-occurrence can't downgrade name
    cur_section_name = ""
    cur_rule_no = ""
    cur_rule_name = ""
    buf: list[str] = []

    def flush() -> None:
        if not cur_rule_no:
            return
        body = "\n".join(buf).strip()
        if not body:
            return  # empty body == TOC artifact; drop it
        rule_name = title_case_name(cur_rule_name)
        section_header = (
            f"{cur_section} - Rule {cur_rule_no} {rule_name}".strip(" -")
            if cur_section
            else f"Rule {cur_rule_no} {rule_name}".strip()
        )
        for piece in _recursive_split(
            body, config.MAX_CHUNK_CHARS, config.SUBCHUNK_OVERLAP
        ):
            chunks.append(
                Chunk(
                    league=league,
                    rule_number=cur_rule_no,
                    rule_name=rule_name,
                    section_header=section_header,
                    text=piece.strip(),
                )
            )

    i, n = 0, len(lines)
    while i < n:
        line = lines[i]

        sec_m = spec.section_re.match(line)
        if sec_m:
            num = _norm_section_no(sec_m.group(1))
            inline_sname = _optional_group(sec_m, "sname")
            sname = inline_sname
            if (
                not sname
                and i + 1 < n
                and not _is_header(lines[i + 1], spec)
                and _looks_like_name(lines[i + 1])
            ):
                sname = lines[i + 1]
                i += 1
            # Don't let a bare re-occurrence of the same section overwrite a good
            # name we already captured (e.g. IIHF "SECTION 01" divider pages).
            if num == cur_section_no and not inline_sname:
                sname = cur_section_name or sname
            cur_section_no = num
            cur_section_name = sname
            cur_section = f"Section {num} - {sname}".strip(" -")
            i += 1
            continue

        rule_m = spec.rule_re.match(line)
        if rule_m:
            inline_name = _optional_group(rule_m, "name")
            # A header that carries an inline name which reads like a sentence
            # (long, or with a period) is a cross-reference wrapped to the start
            # of a line, e.g. "Rule 70 – Leaving the Bench. The player...".
            # Treat it as body text, not a new chunk boundary.
            if inline_name and not _looks_like_name(inline_name):
                if cur_rule_no:
                    buf.append(line)
                i += 1
                continue

            flush()           # close the previous rule
            buf = []
            cur_rule_no = rule_m.group(1).strip()
            cur_rule_name = inline_name
            # name-on-next-line layouts: read the name from the following line,
            # but only if it's a plausible name (guards against swallowing body).
            if (
                not cur_rule_name
                and i + 1 < n
                and not _is_header(lines[i + 1], spec)
                and _looks_like_name(lines[i + 1])
            ):
                cur_rule_name = lines[i + 1]
                i += 1
            i += 1
            continue

        if cur_rule_no:
            buf.append(line)
        i += 1

    flush()  # final rule
    return chunks


def backfill_names(chunks: list[Chunk]) -> list[Chunk]:
    """Fill empty rule_name/section_header from a same-numbered named sibling.

    Some chunks are continuation fragments split off by a cross-reference that
    wrapped to the start of a line; they carry the right rule_number but no
    name. We borrow the name (and section_header) from another chunk of the
    same rule so every chunk stays fully citable.
    """
    name_by_rule: dict[str, str] = {}
    section_by_rule: dict[str, str] = {}
    for c in chunks:
        if c.rule_name and c.rule_number not in name_by_rule:
            name_by_rule[c.rule_number] = c.rule_name
            section_by_rule[c.rule_number] = c.section_header
    for c in chunks:
        if not c.rule_name and c.rule_number in name_by_rule:
            c.rule_name = name_by_rule[c.rule_number]
            c.section_header = section_by_rule[c.rule_number]
    return chunks


# ===========================================================================
# Driver
# ===========================================================================
def ingest_league(league: str) -> list[Chunk]:
    if league not in LEAGUE_SPECS:
        raise NotImplementedError(
            f"No parsing spec for league '{league}' yet. "
            f"NHL is implemented; others are tuned in a later Phase-2 pass."
        )
    spec = LEAGUE_SPECS[league]
    pdf_name = config.RAW_PDF_FILES[league]
    pdf_path = config.RAW_DIR / pdf_name
    if not pdf_path.exists():
        raise FileNotFoundError(f"Expected rulebook at {pdf_path}")

    print(f"[ingest] {league}: extracting text from {pdf_name} ...")
    pages = extract_pages(pdf_path)
    lines = clean_lines(pages, spec)
    print(f"[ingest] {league}: {len(pages)} pages -> {len(lines)} content lines")
    chunks = chunk_lines(lines, league, spec)
    chunks = backfill_names(chunks)
    print(f"[ingest] {league}: produced {len(chunks)} chunks")
    return chunks


def write_jsonl(chunks: list[Chunk], league: str) -> Path:
    config.CHUNKS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.CHUNKS_DIR / f"{league}.jsonl"
    with out_path.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")
    print(f"[ingest] {league}: wrote {len(chunks)} chunks -> {out_path}")
    return out_path


def print_samples(chunks: list[Chunk], n: int = 3) -> None:
    """Show a few representative chunks for human eyeballing."""
    # Pick some by rule number if present, else just the first few.
    wanted = {"81", "78", "16", "20", "22", "23"}  # icing, offside, penalties
    picks = [c for c in chunks if c.rule_number in wanted][:n]
    if len(picks) < n:
        picks += chunks[: n - len(picks)]
    print("\n" + "=" * 70)
    print(f"SAMPLE CHUNKS ({len(picks)} of {len(chunks)})")
    print("=" * 70)
    for c in picks:
        print(f"\nleague:         {c.league}")
        print(f"rule_number:    {c.rule_number}")
        print(f"rule_name:      {c.rule_name}")
        print(f"section_header: {c.section_header}")
        preview = c.text[:400].replace("\n", " ")
        print(f"text[:400]:     {preview}{'...' if len(c.text) > 400 else ''}")
        print("-" * 70)


def main() -> None:
    ap = argparse.ArgumentParser(description="QuickWhistle ingestion")
    ap.add_argument("--league", help="League code, e.g. NHL")
    ap.add_argument("--all", action="store_true", help="Ingest all configured leagues")
    ap.add_argument(
        "--sample",
        action="store_true",
        help="Print sample chunks and do NOT write the JSONL (review mode)",
    )
    args = ap.parse_args()

    if args.all:
        targets = list(config.LEAGUES)
    elif args.league:
        targets = [args.league.upper()]
    else:
        ap.error("Provide --league NHL or --all")

    for league in targets:
        try:
            chunks = ingest_league(league)
        except NotImplementedError as e:
            print(f"[skip] {e}")
            continue
        if args.sample:
            print_samples(chunks)
        else:
            write_jsonl(chunks, league)
            print_samples(chunks)


if __name__ == "__main__":
    main()
