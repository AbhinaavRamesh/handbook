#!/usr/bin/env python3
"""Generate an Anki deck (.apkg) from The Handbook's ML interview FAQ.

The FAQ pages under docs/ml-fundamentals/interview-faq/ are already written as
Q&A: each question is a `### Q<n>: ...` heading and the answer is everything up
to the next heading. We turn each (question, answer) pair into a Basic Anki note.

Limitations (intentional, to keep the deck self-contained and the package simple):
  - Images, Mermaid diagrams, and VitePress ::: containers are stripped.
  - Block math is dropped; inline math is left as written.
Those richer assets live on the site itself.

Regenerate:
    pip install genanki markdown
    python docs/scripts/generate_anki_deck.py

Output: docs/public/downloads/handbook-ml-interview-faq.apkg
"""

import re
import sys
from pathlib import Path

try:
    import genanki
except ImportError:
    sys.exit("Missing dependency: pip install genanki markdown")

try:
    import markdown as md_lib
except ImportError:
    sys.exit("Missing dependency: pip install genanki markdown")

# Stable, hardcoded IDs so regeneration is deterministic (never randomize).
DECK_ID = 1672004812
MODEL_ID = 1672004813
DECK_NAME = "The Handbook — ML Interview FAQ"

ROOT = Path(__file__).resolve().parents[2]
FAQ_DIR = ROOT / "docs" / "ml-fundamentals" / "interview-faq"
OUT_PATH = ROOT / "docs" / "public" / "downloads" / "handbook-ml-interview-faq.apkg"
SITE = "https://abhinaavramesh.github.io/handbook"

QUESTION_RE = re.compile(r"^###\s+Q\d+[:.\)]?\s*(.+?)\s*$")
ANY_HEADING_RE = re.compile(r"^#{1,6}\s")
IMAGE_RE = re.compile(r"!\[[^\]]*\]\([^)]*\)")
CONTAINER_RE = re.compile(r"^:::.*$")
MERMAID_FENCE = "```mermaid"

MODEL = genanki.Model(
    MODEL_ID,
    "Handbook Basic",
    fields=[{"name": "Front"}, {"name": "Back"}, {"name": "Topic"}],
    templates=[
        {
            "name": "Q&A",
            "qfmt": '<div class="topic">{{Topic}}</div><div class="q">{{Front}}</div>',
            "afmt": '{{FrontSide}}<hr id="answer"><div class="a">{{Back}}</div>',
        }
    ],
    css=(
        ".card{font-family:-apple-system,Segoe UI,Roboto,sans-serif;font-size:16px;"
        "line-height:1.5;color:#1e293b;background:#fff;max-width:680px;margin:0 auto;"
        "padding:16px;text-align:left}"
        ".topic{font-size:12px;text-transform:uppercase;letter-spacing:.05em;"
        "color:#2563eb;margin-bottom:8px}"
        ".q{font-weight:600;font-size:18px}"
        ".a pre{background:#f1f5f9;padding:10px;border-radius:6px;overflow-x:auto}"
        ".a code{background:#f1f5f9;padding:1px 4px;border-radius:4px}"
        ".a table{border-collapse:collapse}.a th,.a td{border:1px solid #cbd5e1;padding:4px 8px}"
        "hr#answer{border:none;border-top:1px solid #e2e8f0;margin:14px 0}"
    ),
)


def clean_markdown(block: str) -> str:
    """Drop images, Mermaid blocks, and ::: containers; keep prose/code/tables."""
    out, in_mermaid = [], False
    for line in block.splitlines():
        stripped = line.strip()
        if stripped.startswith(MERMAID_FENCE):
            in_mermaid = True
            continue
        if in_mermaid:
            if stripped.startswith("```"):
                in_mermaid = False
            continue
        if CONTAINER_RE.match(stripped):  # ::: tip / ::: — keep inner text, drop marker
            continue
        line = IMAGE_RE.sub("", line)
        out.append(line)
    return "\n".join(out).strip()


def title_of(path: Path) -> str:
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].replace("- Interview FAQ", "").replace("Interview FAQ", "").strip(" -")
    return path.stem.replace("-", " ").title()


def parse_file(path: Path):
    lines = path.read_text(encoding="utf-8").splitlines()
    topic = title_of(path)
    cards, i, n = [], 0, len(lines)
    while i < n:
        m = QUESTION_RE.match(lines[i])
        if not m:
            i += 1
            continue
        question = m.group(1).strip()
        body, i = [], i + 1
        while i < n and not ANY_HEADING_RE.match(lines[i]):
            body.append(lines[i])
            i += 1
        answer_md = clean_markdown("\n".join(body))
        if question and answer_md:
            cards.append((topic, question, answer_md))
    return cards


def main():
    if not FAQ_DIR.is_dir():
        sys.exit(f"FAQ dir not found: {FAQ_DIR}")

    md = md_lib.Markdown(extensions=["fenced_code", "tables", "sane_lists"])
    deck = genanki.Deck(DECK_ID, DECK_NAME)

    files = sorted(FAQ_DIR.rglob("*.md"))
    total, topics = 0, set()
    for path in files:
        if path.name == "index.md":
            continue
        for topic, question, answer_md in parse_file(path):
            md.reset()
            answer_html = md.convert(answer_md)
            note = genanki.Note(
                model=MODEL,
                fields=[question, answer_html, topic],
                tags=["handbook", re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-")],
            )
            deck.add_note(note)
            total += 1
            topics.add(topic)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    genanki.Package(deck).write_to_file(str(OUT_PATH))

    print(f"Wrote {total} cards across {len(topics)} topics")
    print(f"Output: {OUT_PATH.relative_to(ROOT)}")
    print(f"Source: {FAQ_DIR.relative_to(ROOT)}  ({SITE}/ml-fundamentals/interview-faq/)")


if __name__ == "__main__":
    main()
