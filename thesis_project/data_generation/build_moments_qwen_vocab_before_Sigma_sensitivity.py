"""
Build a Qwen same-token-length vocabulary from every MOMENTS transcript JSON.

This script scans the full MOMENTS transcript tree, not just the curated task
annotations, and collects replacement words from the `local` and `global`
transcript fields.

The output preserves the legacy flat `buckets` mapping used by the current
dataset builder, and adds `pos_buckets` so the replacement pool is also
organized by coarse POS category (noun, verb, adjective, etc.).
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Optional, Set, Tuple

from functools import lru_cache

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MOMENTS_ROOT = REPO_ROOT / "thesis_project" / "data" / "MOMENTS"
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT
    / "reproducing_code"
    / "vlm-circuits-analysis"
    / "data"
    / "moments_goal"
    / "qwen_same_token_length_vocab.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--moments_root",
        default=str(DEFAULT_MOMENTS_ROOT),
        help="Root directory containing all MOMENTS clip folders.",
    )
    parser.add_argument(
        "--tokenizer_path",
        required=True,
        help="HF model path used to measure exact Qwen token lengths.",
    )
    parser.add_argument(
        "--output_path",
        default=str(DEFAULT_OUTPUT_PATH),
        help="Destination JSON file for the vocab cache.",
    )
    return parser.parse_args()


@lru_cache(maxsize=4)
def load_tokenizer(tokenizer_path: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)


@lru_cache(maxsize=1)
def load_spacy_nlp():
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError("spaCy is required to build the MOMENTS vocab") from exc

    try:
        return spacy.load("en_core_web_sm")
    except OSError as exc:
        raise RuntimeError(
            "Missing spaCy model en_core_web_sm. Install it before building the vocab."
        ) from exc


def token_length(tokenizer, text: str) -> int:
    if not text or not text.strip():
        return 0
    encoded = tokenizer(text, add_special_tokens=False)
    return len(encoded["input_ids"])


def is_candidate(token) -> bool:
    if token.is_space or token.is_punct:
        return False
    surface = token.text.strip()
    if not surface:
        return False
    return any(ch.isalpha() for ch in surface)


def normalize_surface(text: str) -> str:
    return text.strip().lower()


def normalize_pos(token) -> str:
    coarse = token.pos_ or "X"
    if coarse in {"NOUN", "PROPN", "VERB", "ADJ", "ADV"}:
        return coarse
    if coarse in {"AUX"}:
        return "VERB"
    return "OTHER"


def iter_transcript_jsons(moments_root: Path) -> Iterable[Path]:
    for path in sorted(moments_root.rglob("*.json")):
        if path.is_file():
            yield path


def extract_transcript_texts(data: object) -> List[str]:
    if not isinstance(data, dict):
        return []
    texts: List[str] = []
    for field in ("local", "global"):
        text = str(data.get(field, "")).strip()
        if text:
            texts.append(text)
    return texts


def add_token(
    token,
    *,
    tokenizer,
    flat_buckets: DefaultDict[str, Set[str]],
    pos_buckets: DefaultDict[str, DefaultDict[str, Set[str]]],
) -> None:
    if not is_candidate(token):
        return
    surface = normalize_surface(token.text)
    if not surface:
        return
    length = token_length(tokenizer, surface)
    if length <= 0:
        return
    pos = normalize_pos(token)
    flat_buckets[str(length)].add(surface)
    pos_buckets[pos][str(length)].add(surface)


def build_vocab(moments_root: Path, tokenizer_path: str) -> Dict[str, object]:
    moments_root = moments_root.resolve()
    tokenizer = load_tokenizer(tokenizer_path)
    nlp = load_spacy_nlp()

    flat_buckets: DefaultDict[str, Set[str]] = defaultdict(set)
    pos_buckets: DefaultDict[str, DefaultDict[str, Set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )

    n_json_seen = 0
    n_text_fields_seen = 0
    for json_path in iter_transcript_jsons(moments_root):
        with json_path.open("r") as f:
            data = json.load(f)
        texts = extract_transcript_texts(data)
        if not texts:
            continue
        n_json_seen += 1
        for text in texts:
            n_text_fields_seen += 1
            doc = nlp(text)
            for token in doc:
                add_token(
                    token,
                    tokenizer=tokenizer,
                    flat_buckets=flat_buckets,
                    pos_buckets=pos_buckets,
                )

    buckets_out = {
        bucket: sorted(words) for bucket, words in sorted(flat_buckets.items(), key=lambda item: int(item[0]))
    }
    pos_buckets_out = {
        pos: {
            bucket: sorted(words)
            for bucket, words in sorted(length_map.items(), key=lambda item: int(item[0]))
        }
        for pos, length_map in sorted(pos_buckets.items())
    }

    return {
        "tokenizer_path": tokenizer_path,
        "moments_root": str(moments_root),
        "n_json_files_used": n_json_seen,
        "n_text_fields_used": n_text_fields_seen,
        "buckets": buckets_out,
        "pos_buckets": pos_buckets_out,
    }


def main() -> None:
    args = parse_args()
    moments_root = Path(args.moments_root)
    if not moments_root.exists():
        raise FileNotFoundError(f"Missing MOMENTS root: {moments_root}")

    payload = build_vocab(moments_root, args.tokenizer_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)

    print(f"Saved vocab cache to {output_path}")
    print(
        f"Used {payload['n_json_files_used']} transcript JSON files and {payload['n_text_fields_used']} text fields"
    )


if __name__ == "__main__":
    main()
