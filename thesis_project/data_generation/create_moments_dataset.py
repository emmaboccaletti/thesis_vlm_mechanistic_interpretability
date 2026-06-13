"""
Create MOMENTS dataset CSVs for the VLM circuit-analysis pipeline.

This script converts the MOMENTS annotations and extracted frame sequences into
pipeline-ready CSV files with explicit clean/counterfactual fields.

It supports two MOMENTS task variants:
  - goal
  - important

For each task variant, it writes four CSVs:
  - random_pair
  - language_only
  - vision_only
  - both

The clean sample is always a single horizontal composite image built from the
ordered frame sequence.
The perturbation-based counterfactual uses the same clip representation with
visual Gaussian noise applied to the constituent frames before composition.
If extracted frames are unavailable for a clip, the builder falls back to
sampling the mp4 directly and composes the strip from those sampled frames.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import json
import os
from pathlib import Path
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image, ImageFile, ImageOps

ImageFile.LOAD_TRUNCATED_IMAGES = True

REPO_ROOT = Path(__file__).resolve().parents[2]


QUESTION_BY_TASK = {
    "goal": "Is this a goal? Answer yes or no.",
    "important": "Is this an important moment in the match? Answer yes or no.",
    "event_type": "What type of moment is this? Answer goal, corner, or shot.",
}

ANNOTATION_BY_TASK = {
    # Use the broader category annotations for the goal task so NIMs can serve
    # as negative examples and the local full run is not limited to IMs only.
    "goal": "category_annotation.csv",
    "important": "category_annotation.csv", # The category annotations include both important and non-important clips, labeled "important" and "not_important". It is also divided into 3 categories (event_type): GOAL, CORNER/THROW-IN, SHOT-ON-TARGET
    "event_type": "category_annotation.csv", # The category annotations include the three event types: GOAL, CORNER/THROW-IN, SHOT-ON-TARGET.
}

EVENT_TYPE_TO_ANSWER = {
    "GOAL": "goal",
    "CORNER/THROW-IN": "corner",
    "SHOT-ON-TARGET": "shot",
}

EVENT_TYPE_ANSWER_ORDER = ["goal", "corner", "shot"]
GOAL_TASK_EVENT_PRIORITY = {
    "GOAL": 3,
    "SHOT-ON-TARGET": 2,
    "CORNER/THROW-IN": 1,
}

# A small curated lexicon based on the local MOMENTS transcript JSONs. These
# replacements are football-domain friendly and are preferred over the generic
# WordNet antonym path when available.
FOOTBALL_DOMAIN_ADJECTIVE_SUBSTITUTIONS = {
    "wonderful": ["terrible", "awful"],
    "brilliant": ["terrible", "awful", "poor"],
    "lovely": ["awful", "terrible"],
    "fantastic": ["terrible", "awful"],
    "spectacular": ["ordinary", "plain"],
    "rampant": ["tame", "contained", "subdued"],
    "good": ["bad", "poor"],
    "great": ["poor", "bad"],
    "bad": ["good", "great"],
    "big": ["small"],
    "little": ["big", "large"],
    "quick": ["slow"],
    "long": ["short"],
    "high": ["low"],
    "clear": ["unclear", "muddy"],
    "comfortable": ["uncomfortable"],
    "dangerous": ["safe"],
    "careful": ["reckless"],
    "risky": ["safe"],
    "decent": ["poor"],
    "nice": ["awful"],
    "fine": ["poor"],
    "open": ["closed"],
    "bright": ["dull"],
    "easy": ["hard"],
    "clean": ["messy"],
    "right": ["wrong"],
    "near": ["far"],
    "far": ["near"],
    "sure": ["unsure"],
    "final": ["early"],
    "second": ["first"],
    "opposite": ["same"],
}

NEGATION_CONTRACTIONS = {
    "be": {
        "is": "isn't",
        "are": "aren't",
        "was": "wasn't",
        "were": "weren't",
    },
    "have": {
        "have": "haven't",
        "has": "hasn't",
        "had": "hadn't",
    },
    "do": {
        "do": "don't",
        "does": "doesn't",
        "did": "didn't",
    },
}

CSV_COLUMNS = [
    "clip_id",
    "group_idx",
    "clip_name",
    "event_type",
    "label",
    "similarity",
    "local_text",
    "global_text",
    "prompt",
    "image_paths",
    "answer",
    "cf_mode",
    "cf_prompt",
    "cf_image_paths",
    "cf_answer",
    "cf_prompt_state",
    "cf_prompt_changes",
]

COMPOSITE_TILE_SIZE = 244
COMPOSITE_SEPARATOR_PX = 4
COMPOSITE_BACKGROUND = (255, 255, 255)


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_annotations_dir = repo_root / "thesis_project" / "data" / "MOMENTS_categories"
    default_moments_root = repo_root / "thesis_project" / "data" / "MOMENTS"
    default_frames_root = repo_root / "thesis_project" / "data" / "MOMENTS_frames" / "frames"
    default_output_dir = repo_root / "reproducing_code" / "vlm-circuits-analysis" / "data"

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--task",
        choices=("goal", "important", "event_type"),
        default="goal",
        help="Which MOMENTS task to generate.",
    )
    parser.add_argument(
        "--annotations_dir",
        default=str(default_annotations_dir),
        help="Directory containing goals_annotation_only_IM.csv / category_annotation.csv.",
    )
    parser.add_argument(
        "--moments_root",
        default=str(default_moments_root),
        help="Root directory of the raw MOMENTS video data.",
    )
    parser.add_argument(
        "--frames_root",
        default=str(default_frames_root),
        help="Root directory containing extracted frames.",
    )
    parser.add_argument(
        "--output_dir",
        default=str(default_output_dir),
        help="Output directory for the generated MOMENTS task data.",
    )
    parser.add_argument(
        "--similarity_threshold",
        type=float,
        default=0, #default=0.5,
        help="Skip clips whose transcript similarity is below this threshold.",
    )
    parser.add_argument(
        "--n_frames",
        type=int,
        default=10,
        help="Target number of frames per clip.",
    )
    parser.add_argument(
        "--noise_sigma",
        type=float,
        default=0.12,
        help="Standard deviation of Gaussian noise for vision corruptions.",
    )
    parser.add_argument(
        "--language_perturbation_fraction",
        type=float,
        default=0.10,
        help=(
            "Fraction of perturbable content words (ADJ/VERB/NOUN that can be "
            "rewritten) to change in each sentence for language counterfactuals."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="If > 0, only process the first N annotation rows for a local subset run.",
    )
    parser.add_argument(
        "--no_write_noisy_frames",
        action="store_true",
        help="Do not write noisy PNGs; only emit CSV rows that point to the expected noisy-frame paths.",
    )
    parser.add_argument(
        "--noisy_frames_root",
        default="",
        help=(
            "Optional root directory to use when constructing noisy-frame paths. "
            "Useful when you want the CSVs to point to an existing noisy-frame tree."
        ),
    )
    parser.add_argument(
        "--prefer_mp4",
        action="store_true",
        help=(
            "Prefer sampling frames directly from the source mp4 instead of "
            "using extracted PNG frames when available."
        ),
    )
    return parser.parse_args()


def parse_annotation_entry(entry: str) -> Tuple[str, str, str]:
    """
    Parse an annotation key of the form clip_id-group_idx-clip_name.

    clip_ids may contain hyphens, so split from the right.
    """
    parts = entry.rsplit("-", 2)
    if len(parts) != 3:
        raise ValueError(f"Could not parse annotation entry: {entry}")
    return parts[0], parts[1], parts[2]


def load_annotation_rows(annotations_dir: str, task: str) -> List[Dict[str, str]]:
    path = Path(annotations_dir) / ANNOTATION_BY_TASK[task]
    if not path.exists():
        raise FileNotFoundError(f"Missing annotation CSV: {path}")

    rows: List[Dict[str, str]] = []
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        rows.extend(reader)
    return rows


def deduplicate_goal_annotations(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Keep one annotation row per mp4_path for the goal task.

    Priority: GOAL > SHOT-ON-TARGET > CORNER/THROW-IN.
    """
    best_rows: Dict[str, Dict[str, str]] = {}
    best_priority: Dict[str, int] = {}
    for row in rows:
        mp4_path = row["mp4_path"]
        event_type = row["event_type"].strip().upper()
        priority = GOAL_TASK_EVENT_PRIORITY.get(event_type, 0)
        if mp4_path not in best_rows or priority > best_priority[mp4_path]:
            best_rows[mp4_path] = row
            best_priority[mp4_path] = priority
    return list(best_rows.values())


def resolve_path(path: str, repo_root: Optional[str] = None) -> str:
    """
    Resolve MOMENTS paths across the cluster and local checkout.

    The annotations often store /home/eboccaletti/... paths, while the local
    checkout lives under /Users/e.boccaletti/....
    """
    if os.path.exists(path):
        return path

    if repo_root is None:
        repo_root = str(Path(__file__).resolve().parents[2])

    replacements = [
        (
            "/home/eboccaletti/thesis_project/data/MOMENTS",
            os.path.join(repo_root, "thesis_project", "data", "MOMENTS"),
        ),
        (
            "/home/eboccaletti/thesis_project/data/MOMENTS_frames/frames",
            os.path.join(repo_root, "thesis_project", "data", "MOMENTS_frames", "frames"),
        ),
        (
            "/home/eboccaletti/thesis_project/data/MOMENTS_frames",
            os.path.join(repo_root, "thesis_project", "data", "MOMENTS_frames"),
        ),
    ]
    for old, new in replacements:
        if path.startswith(old):
            candidate = path.replace(old, new, 1)
            if os.path.exists(candidate):
                return candidate
    return path


def resolve_transcript_path(mp4_path: str) -> Optional[str]:
    """
    Resolve the sibling JSON file for a given MP4 path.

    MOMENTS files are commonly named _v1.json or _v2.json; we try both and fall
    back to a plain .json sibling if needed.
    """
    mp4_path = resolve_path(mp4_path)
    candidate_paths = [
        resolve_path(mp4_path.replace(".mp4", "_v1.json")),
        resolve_path(mp4_path.replace(".mp4", "_v2.json")),
        resolve_path(mp4_path.replace(".mp4", ".json")),
    ]
    for path in candidate_paths:
        if os.path.exists(path):
            return path
    return None


def load_transcript_data(mp4_path: str) -> Optional[Dict[str, object]]:
    transcript_path = resolve_transcript_path(mp4_path)
    if transcript_path is None:
        return None
    with open(transcript_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Transcript file {transcript_path} did not contain a JSON object")
    return data


@lru_cache(maxsize=1)
def load_spacy_nlp():
    """
    Load the spaCy English pipeline once.

    We only use this for the language-only perturbation path. If the model is
    unavailable locally, we fall back to the global transcript text.
    """
    try:
        import spacy
    except ImportError:
        return None

    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        return None


def get_wordnet_antonyms(word: str) -> List[str]:
    try:
        from nltk.corpus import wordnet as wn
    except ImportError:
        return []

    antonyms: List[str] = []
    try:
        for synset in wn.synsets(word):
            for lemma in synset.lemmas():
                for antonym in lemma.antonyms():
                    antonyms.append(antonym.name().replace("_", " "))
    except LookupError:
        return []

    seen = set()
    ordered: List[str] = []
    for antonym in antonyms:
        lowered = antonym.lower()
        if lowered not in seen and lowered != word.lower():
            seen.add(lowered)
            ordered.append(antonym)
    return ordered


def wordnet_pos_for_spacy_pos(pos: str):
    try:
        from nltk.corpus import wordnet as wn
    except ImportError:
        return None

    if pos == "ADJ":
        return wn.ADJ
    if pos == "VERB":
        return wn.VERB
    if pos == "NOUN":
        return wn.NOUN
    return None


IRREGULAR_VERB_FORMS = {
    "be": {"VBZ": "is", "VBD": "was", "VBN": "been", "VBG": "being"},
    "begin": {"VBZ": "begins", "VBD": "began", "VBN": "begun", "VBG": "beginning"},
    "bring": {"VBZ": "brings", "VBD": "brought", "VBN": "brought", "VBG": "bringing"},
    "build": {"VBZ": "builds", "VBD": "built", "VBN": "built", "VBG": "building"},
    "buy": {"VBZ": "buys", "VBD": "bought", "VBN": "bought", "VBG": "buying"},
    "catch": {"VBZ": "catches", "VBD": "caught", "VBN": "caught", "VBG": "catching"},
    "choose": {"VBZ": "chooses", "VBD": "chose", "VBN": "chosen", "VBG": "choosing"},
    "come": {"VBZ": "comes", "VBD": "came", "VBN": "come", "VBG": "coming"},
    "do": {"VBZ": "does", "VBD": "did", "VBN": "done", "VBG": "doing"},
    "drink": {"VBZ": "drinks", "VBD": "drank", "VBN": "drunk", "VBG": "drinking"},
    "drive": {"VBZ": "drives", "VBD": "drove", "VBN": "driven", "VBG": "driving"},
    "eat": {"VBZ": "eats", "VBD": "ate", "VBN": "eaten", "VBG": "eating"},
    "fall": {"VBZ": "falls", "VBD": "fell", "VBN": "fallen", "VBG": "falling"},
    "feel": {"VBZ": "feels", "VBD": "felt", "VBN": "felt", "VBG": "feeling"},
    "find": {"VBZ": "finds", "VBD": "found", "VBN": "found", "VBG": "finding"},
    "fly": {"VBZ": "flies", "VBD": "flew", "VBN": "flown", "VBG": "flying"},
    "forget": {"VBZ": "forgets", "VBD": "forgot", "VBN": "forgotten", "VBG": "forgetting"},
    "forgive": {"VBZ": "forgives", "VBD": "forgave", "VBN": "forgiven", "VBG": "forgiving"},
    "get": {"VBZ": "gets", "VBD": "got", "VBN": "gotten", "VBG": "getting"},
    "give": {"VBZ": "gives", "VBD": "gave", "VBN": "given", "VBG": "giving"},
    "go": {"VBZ": "goes", "VBD": "went", "VBN": "gone", "VBG": "going"},
    "grow": {"VBZ": "grows", "VBD": "grew", "VBN": "grown", "VBG": "growing"},
    "have": {"VBZ": "has", "VBD": "had", "VBN": "had", "VBG": "having"},
    "hear": {"VBZ": "hears", "VBD": "heard", "VBN": "heard", "VBG": "hearing"},
    "hide": {"VBZ": "hides", "VBD": "hid", "VBN": "hidden", "VBG": "hiding"},
    "hold": {"VBZ": "holds", "VBD": "held", "VBN": "held", "VBG": "holding"},
    "keep": {"VBZ": "keeps", "VBD": "kept", "VBN": "kept", "VBG": "keeping"},
    "know": {"VBZ": "knows", "VBD": "knew", "VBN": "known", "VBG": "knowing"},
    "lead": {"VBZ": "leads", "VBD": "led", "VBN": "led", "VBG": "leading"},
    "leave": {"VBZ": "leaves", "VBD": "left", "VBN": "left", "VBG": "leaving"},
    "lose": {"VBZ": "loses", "VBD": "lost", "VBN": "lost", "VBG": "losing"},
    "make": {"VBZ": "makes", "VBD": "made", "VBN": "made", "VBG": "making"},
    "mean": {"VBZ": "means", "VBD": "meant", "VBN": "meant", "VBG": "meaning"},
    "meet": {"VBZ": "meets", "VBD": "met", "VBN": "met", "VBG": "meeting"},
    "pay": {"VBZ": "pays", "VBD": "paid", "VBN": "paid", "VBG": "paying"},
    "put": {"VBZ": "puts", "VBD": "put", "VBN": "put", "VBG": "putting"},
    "read": {"VBZ": "reads", "VBD": "read", "VBN": "read", "VBG": "reading"},
    "ride": {"VBZ": "rides", "VBD": "rode", "VBN": "ridden", "VBG": "riding"},
    "ring": {"VBZ": "rings", "VBD": "rang", "VBN": "rung", "VBG": "ringing"},
    "rise": {"VBZ": "rises", "VBD": "rose", "VBN": "risen", "VBG": "rising"},
    "run": {"VBZ": "runs", "VBD": "ran", "VBN": "run", "VBG": "running"},
    "say": {"VBZ": "says", "VBD": "said", "VBN": "said", "VBG": "saying"},
    "see": {"VBZ": "sees", "VBD": "saw", "VBN": "seen", "VBG": "seeing"},
    "send": {"VBZ": "sends", "VBD": "sent", "VBN": "sent", "VBG": "sending"},
    "set": {"VBZ": "sets", "VBD": "set", "VBN": "set", "VBG": "setting"},
    "shake": {"VBZ": "shakes", "VBD": "shook", "VBN": "shaken", "VBG": "shaking"},
    "show": {"VBZ": "shows", "VBD": "showed", "VBN": "shown", "VBG": "showing"},
    "sing": {"VBZ": "sings", "VBD": "sang", "VBN": "sung", "VBG": "singing"},
    "sit": {"VBZ": "sits", "VBD": "sat", "VBN": "sat", "VBG": "sitting"},
    "speak": {"VBZ": "speaks", "VBD": "spoke", "VBN": "spoken", "VBG": "speaking"},
    "spend": {"VBZ": "spends", "VBD": "spent", "VBN": "spent", "VBG": "spending"},
    "stand": {"VBZ": "stands", "VBD": "stood", "VBN": "stood", "VBG": "standing"},
    "swim": {"VBZ": "swims", "VBD": "swam", "VBN": "swum", "VBG": "swimming"},
    "take": {"VBZ": "takes", "VBD": "took", "VBN": "taken", "VBG": "taking"},
    "teach": {"VBZ": "teaches", "VBD": "taught", "VBN": "taught", "VBG": "teaching"},
    "tell": {"VBZ": "tells", "VBD": "told", "VBN": "told", "VBG": "telling"},
    "think": {"VBZ": "thinks", "VBD": "thought", "VBN": "thought", "VBG": "thinking"},
    "throw": {"VBZ": "throws", "VBD": "threw", "VBN": "thrown", "VBG": "throwing"},
    "understand": {"VBZ": "understands", "VBD": "understood", "VBN": "understood", "VBG": "understanding"},
    "wake": {"VBZ": "wakes", "VBD": "woke", "VBN": "woken", "VBG": "waking"},
    "wear": {"VBZ": "wears", "VBD": "wore", "VBN": "worn", "VBG": "wearing"},
    "win": {"VBZ": "wins", "VBD": "won", "VBN": "won", "VBG": "winning"},
    "write": {"VBZ": "writes", "VBD": "wrote", "VBN": "written", "VBG": "writing"},
}


def inflect_antonym(token_text: str, antonym: str, token_tag: str) -> str:
    """
    Best-effort surface-form matching for antonym replacement.

    This is intentionally lightweight: we prefer an antonym substitution and
    only do simple morphology for common English verb/noun endings.
    """
    if not antonym:
        return antonym

    source = token_text
    repl = antonym

    if source.isupper():
        return repl.upper()
    if source.istitle():
        return repl.title()

    irregular = IRREGULAR_VERB_FORMS.get(repl.lower())
    if irregular is not None and token_tag in irregular:
        return irregular[token_tag]

    if token_tag in {"VBZ"}:
        if repl.endswith("y") and len(repl) > 1 and repl[-2] not in "aeiou":
            repl = repl[:-1] + "ies"
        elif repl.endswith(("s", "x", "z", "ch", "sh", "o")):
            repl = repl + "es"
        else:
            repl = repl + "s"
    elif token_tag in {"VBD", "VBN"}:
        if repl.endswith("e"):
            repl = repl + "d"
        else:
            repl = repl + "ed"
    elif token_tag == "VBG":
        if repl.endswith("e") and len(repl) > 1:
            repl = repl[:-1] + "ing"
        else:
            repl = repl + "ing"
    elif token_tag in {"NNS", "NNPS"}:
        if repl.endswith("s"):
            pass
        elif repl.endswith(("s", "x", "z", "ch", "sh")):
            repl = repl + "es"
        else:
            repl = repl + "s"

    return repl


def apply_surface_case(source_text: str, replacement: str) -> str:
    if source_text.isupper():
        return replacement.upper()
    if source_text.istitle():
        return replacement.title()
    return replacement


def build_curated_football_replacement(token) -> Optional[Tuple[str, str, str]]:
    if token.pos_ != "ADJ":
        return None

    lemma = token.lemma_.strip().lower()
    candidates = FOOTBALL_DOMAIN_ADJECTIVE_SUBSTITUTIONS.get(lemma)
    if not candidates:
        return None

    for candidate in candidates:
        replacement = apply_surface_case(token.text, candidate)
        if replacement.lower() != token.text.lower():
            return replacement, token.text, "football_curated"
    return None


def build_negation_rewrite(token) -> Optional[Tuple[str, str, str]]:
    lemma = token.lemma_.strip().lower()
    if lemma not in NEGATION_CONTRACTIONS:
        return None

    # Skip sentences that are already negated.
    if any(tok.dep_ == "neg" or tok.text.lower() in {"not", "n't"} for tok in token.sent):
        return None

    token_lower = token.text.strip().lower()
    replacement = NEGATION_CONTRACTIONS[lemma].get(token_lower)
    if replacement is None:
        return None

    return apply_surface_case(token.text, replacement), token.text, "negation_rewrite"


def _pick_antonym_for_token(token, nlp) -> Optional[Tuple[str, str]]:
    """
    Return (replacement, source_word) for the best antonym candidate, if any.
    """
    target_pos = ("ADJ", "VERB", "NOUN")
    if not token.text.strip() or token.is_stop or token.is_punct or token.is_space:
        return None
    if token.pos_ == "NOUN" and token.tag_ == "NNP":
        return None

    wn_pos = wordnet_pos_for_spacy_pos(token.pos_)
    lemmas = []
    token_lemma = token.lemma_.strip().lower()
    if token_lemma:
        lemmas.append(token_lemma)
    if token.text.strip().lower() not in lemmas:
        lemmas.append(token.text.strip().lower())

    antonyms: List[str] = []
    try:
        from nltk.corpus import wordnet as wn
    except ImportError:
        return None

    for lemma in lemmas:
        synsets = wn.synsets(lemma, pos=wn_pos) if wn_pos is not None else wn.synsets(lemma)
        for synset in synsets:
            for lemma_obj in synset.lemmas():
                for antonym in lemma_obj.antonyms():
                    antonym_name = antonym.name().replace("_", " ")
                    if " " in antonym_name:
                        continue
                    antonyms.append(antonym_name)

    if not antonyms:
        return None

    seen = set()
    ordered_antonyms = []
    for antonym in antonyms:
        lowered = antonym.lower()
        if lowered not in seen and lowered != token.text.lower():
            seen.add(lowered)
            ordered_antonyms.append(antonym)
    if not ordered_antonyms:
        return None

    for antonym in ordered_antonyms:
        replacement = inflect_antonym(token.text, antonym, token.tag_)
        if not replacement or replacement.lower() == token.text.lower():
            continue
        candidate_doc = nlp(replacement)
        if len(candidate_doc) != 1:
            continue
        candidate_token = candidate_doc[0]
        if candidate_token.pos_ != token.pos_:
            continue
        if candidate_token.pos_ not in target_pos:
            continue
        return replacement, token.text

    return None


def _replace_one_content_word_in_sentence(
    sentence,
    perturbation_fraction: float = 0.10,
) -> Optional[Tuple[str, str, str]]:
    return _replace_content_words_in_sentence(sentence, perturbation_fraction)


def _best_replacement_for_token(token, nlp) -> Optional[Tuple[str, str, str]]:
    replacement = build_negation_rewrite(token)
    if replacement is not None:
        return replacement

    replacement = build_curated_football_replacement(token)
    if replacement is not None:
        return replacement

    replacement = _pick_antonym_for_token(token, nlp)
    if replacement is not None:
        repl_text, source_word = replacement
        return repl_text, source_word, "spacy_wordnet_antonym"
    return None


def _replace_content_words_in_sentence(
    sentence,
    perturbation_fraction: float,
) -> Optional[Tuple[str, str, str]]:
    """
    Perturb a fraction of the *eligible* content words in one sentence.

    The fraction is measured over tokens that can actually be rewritten by the
    current language perturbation rules, not over the full token count.
    """
    target_pos = ("ADJ", "VERB", "NOUN")
    nlp = load_spacy_nlp()
    if nlp is None:
        return None

    candidates = [token for token in sentence if token.pos_ in target_pos]
    candidates.sort(key=lambda tok: (target_pos.index(tok.pos_), tok.idx))
    if not candidates:
        return None

    target_count = max(1, int(math.ceil(len(candidates) * perturbation_fraction)))

    replacements: List[Tuple[int, int, str, str, int]] = []
    changes: List[str] = []
    prompt_states: List[str] = []

    for token in candidates:
        if len(replacements) >= target_count:
            break
        replacement = _best_replacement_for_token(token, nlp)
        if replacement is None:
            continue
        repl_text, source_word, method = replacement
        replacements.append((token.idx, token.idx + len(token.text), repl_text, source_word, token.idx))
        prompt_states.append(method)
        changes.append(f"sentence[{method}]:{source_word}->{repl_text}")

    if not replacements:
        return None

    perturbed_text = sentence.text
    for start, end, repl_text, _source_word, _tok_idx in sorted(
        replacements, key=lambda item: item[0], reverse=True
    ):
        local_start = start - sentence.start_char
        local_end = end - sentence.start_char
        perturbed_text = perturbed_text[:local_start] + repl_text + perturbed_text[local_end:]

    state_order = ["football_curated", "negation_rewrite", "spacy_wordnet_antonym"]
    state_set = set(prompt_states)
    prompt_state = "+".join([state for state in state_order if state in state_set])
    percent_tag = f"{round(perturbation_fraction * 100):02d}pct"
    prompt_state = "+".join([f"coverage_{percent_tag}", prompt_state] if prompt_state else [f"coverage_{percent_tag}"])
    return perturbed_text, prompt_state, "; ".join(changes)


# Sentence-level version kept here for reference in case we ever want to go
# back to per-sentence coverage.
#
# def replace_one_content_word_with_antonym(
#     text: str,
#     perturbation_fraction: float = 0.10,
# ) -> Optional[Tuple[str, str, str]]:
#     nlp = load_spacy_nlp()
#     if nlp is None:
#         return None
#
#     doc = nlp(text)
#     replacements: List[Tuple[int, int, str, str, int]] = []
#     changes: List[str] = []
#     prompt_states: List[str] = []
#     sentences = list(doc.sents)
#     for sent_idx, sentence in enumerate(sentences, start=1):
#         replacement = _replace_content_words_in_sentence(
#             sentence,
#             perturbation_fraction=perturbation_fraction,
#         )
#         if replacement is None:
#             continue
#         repl_text, prompt_state, change_summary = replacement
#         replacements.append((sentence.start_char, sentence.end_char, repl_text, sentence.text, sent_idx))
#         prompt_states.extend([state for state in prompt_state.split("+") if state])
#         changes.append(f"sentence_{sent_idx}[{prompt_state}]:{change_summary}")
#
#     if not replacements:
#         return None
#
#     perturbed_text = text
#     for start, end, repl_text, _source_word, _sent_idx in sorted(
#         replacements, key=lambda item: item[0], reverse=True
#     ):
#         perturbed_text = perturbed_text[:start] + repl_text + perturbed_text[end:]
#
#     state_order = ["football_curated", "negation_rewrite", "spacy_wordnet_antonym"]
#     state_set = set(prompt_states)
#     percent_tag = f"{round(perturbation_fraction * 100):02d}pct"
#     prompt_state = "+".join([f"coverage_{percent_tag}"] + [state for state in state_order if state in state_set])
#     return perturbed_text, prompt_state, "; ".join(changes)
#

def replace_one_content_word_with_antonym(
    text: str,
    perturbation_fraction: float = 0.10,
) -> Optional[Tuple[str, str, str]]:
    """
    Replace a fraction of eligible content words across the whole description.

    The coverage fraction is applied only to perturbable content words
    (adjectives, verbs, and nouns that pass the rewrite checks), not to every
    token in the sentence. The budget is computed globally over the full text
    and then applied in left-to-right order.

    Returns:
        (perturbed_text, prompt_state, change_summary)
        or None if no suitable replacement can be found.
    """
    nlp = load_spacy_nlp()
    if nlp is None:
        return None

    doc = nlp(text)
    sentences = list(doc.sents)
    sentence_index = {sent.start_char: idx for idx, sent in enumerate(sentences, start=1)}
    candidates: List[Tuple[int, int, object, int]] = []
    for token in doc:
        if token.pos_ not in {"ADJ", "VERB", "NOUN"}:
            continue
        sent_idx = sentence_index.get(token.sent.start_char, 1)
        candidates.append((token.pos_, token.idx, token, sent_idx))
    candidates.sort(key=lambda item: ({"ADJ": 0, "VERB": 1, "NOUN": 2}[item[0]], item[1]))
    if not candidates:
        return None

    target_count = max(1, int(math.ceil(len(candidates) * perturbation_fraction)))
    replacements: List[Tuple[int, int, str, str, int]] = []
    changes: List[str] = []
    prompt_states: List[str] = []

    for _pos, _idx, token, sent_idx in candidates:
        if len(replacements) >= target_count:
            break
        replacement = _best_replacement_for_token(token, nlp)
        if replacement is None:
            continue
        repl_text, source_word, method = replacement
        replacements.append((token.idx, token.idx + len(token.text), repl_text, source_word, sent_idx))
        prompt_states.append(method)
        changes.append(f"sentence_{sent_idx}[{method}]:{source_word}->{repl_text}")

    if not replacements:
        return None

    perturbed_text = text
    for start, end, repl_text, _source_word, _sent_idx in sorted(
        replacements, key=lambda item: item[0], reverse=True
    ):
        perturbed_text = perturbed_text[:start] + repl_text + perturbed_text[end:]

    state_order = ["football_curated", "negation_rewrite", "spacy_wordnet_antonym"]
    state_set = set(prompt_states)
    percent_tag = f"{round(perturbation_fraction * 100):02d}pct"
    prompt_state = "+".join([f"coverage_{percent_tag}"] + [state for state in state_order if state in state_set])
    return perturbed_text, prompt_state, "; ".join(changes)


def build_language_counterfactual_text(
    local_text: str,
    global_text: str,
    perturbation_fraction: float,
) -> Tuple[str, str, str]:
    perturbed = replace_one_content_word_with_antonym(
        local_text,
        perturbation_fraction=perturbation_fraction,
    )
    if perturbed is not None:
        perturbed_text, prompt_state, change_summary = perturbed
        return perturbed_text, prompt_state, change_summary
    return global_text or local_text, "global_fallback", "fallback_to_global_text"


def get_task_question(task: str) -> str:
    return QUESTION_BY_TASK[task]


def build_prompt(description: str, task: str) -> str:
    description = description.strip()
    if description and not description.endswith((".", "!", "?")):
        description += "."
    return (
        "You are watching a football match. "
        f"{description} "
        f"{get_task_question(task)}"
    ).strip()


def parse_frame_paths(frame_dir: str, n_frames: int) -> List[str]:
    frame_paths = sorted(
        str(Path(frame_dir) / name)
        for name in os.listdir(frame_dir)
        if name.startswith("frame_") and name.endswith(".png")
    )
    if len(frame_paths) == 0:
        raise FileNotFoundError(f"No frames found in {frame_dir}")

    if len(frame_paths) == n_frames:
        return frame_paths

    indices = np.linspace(0, len(frame_paths) - 1, num=n_frames, dtype=int)
    return [frame_paths[i] for i in indices]


def load_image(path: str) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def image_to_array(image: Image.Image) -> np.ndarray:
    return np.asarray(image).astype(np.float32) / 255.0


def array_to_image(array: np.ndarray) -> Image.Image:
    array = np.clip(array * 255.0, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(array)


def sample_seed(*parts: str) -> int:
    h = hashlib.sha256("::".join(parts).encode("utf-8")).hexdigest()
    return int(h[:16], 16) % (2**32)


def apply_gaussian_noise(image: Image.Image, sigma: float, seed: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    arr = image_to_array(image)
    noise = rng.normal(loc=0.0, scale=sigma, size=arr.shape).astype(np.float32)
    return array_to_image(arr + noise)


def save_noisy_frame(
    image: Image.Image,
    noisy_path: str,
    sigma: float,
    seed: int,
) -> str:
    os.makedirs(os.path.dirname(noisy_path), exist_ok=True)
    noisy_image = apply_gaussian_noise(image, sigma=sigma, seed=seed)
    noisy_image.save(noisy_path)
    return noisy_path


def get_event_label(task: str, label: str, event_type: str) -> str:
    if task == "goal":
        return "goal" if event_type.upper() == "GOAL" else "not_goal"
    if task == "important":
        return "important" if label == "important" else "not_important"
    normalized_event_type = EVENT_TYPE_TO_ANSWER.get(event_type.upper())
    if normalized_event_type is None:
        raise ValueError(f"Unknown MOMENTS event type: {event_type}")
    return normalized_event_type


def get_answer(task: str, label: str, event_type: str) -> str:
    if task in {"goal", "important"}:
        if task == "goal":
            return "yes" if get_event_label(task, label, event_type) == "goal" else "no"
        return "yes" if get_event_label(task, label, event_type) == "important" else "no"
    return get_event_label(task, label, event_type)


def get_cf_answer(task: str, answer: str, sample_id: str, mode: str) -> str:
    if task in {"goal", "important"}:
        return "no" if answer == "yes" else "yes"

    if task == "event_type":
        other_answers = [candidate for candidate in EVENT_TYPE_ANSWER_ORDER if candidate != answer]
        if not other_answers:
            raise ValueError(f"No alternative answer available for {answer}")
        return other_answers[sample_seed(task, mode, sample_id) % len(other_answers)]

    raise ValueError(f"Unknown MOMENTS task: {task}")


def _fit_frame_into_tile(image: Image.Image, tile_size: int) -> Image.Image:
    """
    Resize a frame while preserving aspect ratio, then center it on a square tile.
    """
    resample = getattr(Image, "Resampling", Image).LANCZOS
    fitted = ImageOps.contain(image, (tile_size, tile_size), method=resample)
    canvas = Image.new("RGB", (tile_size, tile_size), COMPOSITE_BACKGROUND)
    offset = ((tile_size - fitted.width) // 2, (tile_size - fitted.height) // 2)
    canvas.paste(fitted, offset)
    return canvas


def build_horizontal_composite(
    *,
    frame_paths: Optional[Sequence[str]] = None,
    frame_images: Optional[Sequence[Image.Image]] = None,
    output_path: str,
    noise_sigma: Optional[float] = None,
    seed_prefix: Sequence[str] = (),
    tile_size: int = COMPOSITE_TILE_SIZE,
    separator_px: int = COMPOSITE_SEPARATOR_PX,
    write_image: bool = True,
) -> str:
    """
    Build one ordered horizontal strip from a clip's frames.

    If noise_sigma is provided, Gaussian noise is applied to each frame before
    composition. The same composite is used as the single visual input for the
    clip.
    """
    if frame_paths is None and frame_images is None:
        raise ValueError("frame_paths or frame_images must be provided")

    if frame_images is None:
        if not frame_paths:
            raise ValueError("frame_paths must be non-empty")
        frame_images = [load_image(frame_path) for frame_path in frame_paths]
    elif not frame_images:
        raise ValueError("frame_images must be non-empty")

    tiles: List[Image.Image] = []
    for idx, img in enumerate(frame_images):
        if noise_sigma is not None:
            seed = sample_seed(*seed_prefix, str(idx))
            img = apply_gaussian_noise(img, sigma=noise_sigma, seed=seed)
        tiles.append(_fit_frame_into_tile(img, tile_size))

    width = len(tiles) * tile_size + max(0, len(tiles) - 1) * separator_px
    composite = Image.new("RGB", (width, tile_size), COMPOSITE_BACKGROUND)
    for idx, tile in enumerate(tiles):
        x = idx * (tile_size + separator_px)
        composite.paste(tile, (x, 0))

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if write_image and not os.path.exists(output_path):
        composite.save(output_path)
    elif write_image:
        # Preserve the existing artifact if it is already present.
        pass
    return output_path


def build_shared_vision_noisy_composite_path(
    *,
    task: str,
    clip_id: str,
    clip_type: str,
    group_idx: str,
    clip_name: str,
    frame_paths: Optional[Sequence[str]] = None,
    frame_images: Optional[Sequence[Image.Image]] = None,
    output_dir: str,
    noise_sigma: float,
    write_noisy_frames: bool = True,
    noisy_frames_root: str = "",
) -> str:
    """
    Build a shared noisy version of the clean composite strip.

    The same noisy frames are reused by `vision_only` and `both`, so those two
    CSVs can point to identical corrupted composites.
    """
    sigma_tag = str(noise_sigma).replace(".", "p")
    noisy_base = Path(noisy_frames_root) if noisy_frames_root else Path(output_dir)
    noisy_path = (
        noisy_base
        / task
        / f"composites_noisy_sigma{sigma_tag}"
        / clip_id
        / clip_type
        / group_idx
        / f"{clip_name}.png"
    )
    if write_noisy_frames and not noisy_path.exists():
        build_horizontal_composite(
            frame_paths=frame_paths,
            frame_images=frame_images,
            output_path=str(noisy_path),
            noise_sigma=noise_sigma,
            seed_prefix=(task, clip_id, clip_type, group_idx, clip_name),
            write_image=True,
        )
    return str(noisy_path)


def build_clean_composite_path(
    *,
    task: str,
    clip_id: str,
    clip_type: str,
    group_idx: str,
    clip_name: str,
    frame_paths: Optional[Sequence[str]] = None,
    frame_images: Optional[Sequence[Image.Image]] = None,
    output_dir: str,
    write_image: bool = True,
) -> str:
    clean_path = (
        Path(output_dir)
        / task
        / "composites"
        / clip_id
        / clip_type
        / group_idx
        / f"{clip_name}.png"
    )
    if write_image and not clean_path.exists():
        build_horizontal_composite(
            frame_paths=frame_paths,
            frame_images=frame_images,
            output_path=str(clean_path),
            noise_sigma=None,
            seed_prefix=(),
            write_image=True,
        )
    return str(clean_path)


def sample_frame_images_from_mp4(mp4_path: str, n_frames: int) -> List[Image.Image]:
    """
    Sample uniformly spaced frames directly from a MOMENTS mp4.

    This is used as a fallback when extracted PNG frames are not available for a
    clip.
    """
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on local env
        raise ImportError(
            "OpenCV is required to sample frames directly from mp4 files"
        ) from exc

    cap = cv2.VideoCapture(mp4_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        raise FileNotFoundError(f"Could not read any frames from {mp4_path}")

    actual_n = min(n_frames, total)
    indices = np.linspace(0, total - 1, actual_n, dtype=int)

    images: List[Image.Image] = []
    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        images.append(Image.fromarray(frame_rgb))

    cap.release()
    if not images:
        raise FileNotFoundError(f"Could not sample any frames from {mp4_path}")
    return images


def load_clip_visual_inputs(
    *,
    frame_dir: str,
    mp4_path: str,
    n_frames: int,
    prefer_mp4: bool = False,
) -> Tuple[List[Image.Image], str, List[str]]:
    """
    Load clip frames from extracted PNGs when available, otherwise sample the mp4.
    """
    if not prefer_mp4 and os.path.isdir(frame_dir):
        try:
            frame_paths = parse_frame_paths(frame_dir, n_frames=n_frames)
            return [load_image(path) for path in frame_paths], "frames", frame_paths
        except (FileNotFoundError, OSError, ValueError):
            pass
    return sample_frame_images_from_mp4(mp4_path, n_frames=n_frames), "mp4", []


def build_frame_root(
    frames_root: str, mp4_path: str
) -> Tuple[str, str, str, str, str]:
    """
    Build the extracted-frame directory for a raw MOMENTS MP4 path.

    Expected structure:
      {frames_root}/{clip_id}/{im|nim}/{group_idx}/{clip_name}/frame_XX.png
    """
    mp4_path_obj = Path(mp4_path)
    clip_name = mp4_path_obj.stem
    group_idx = mp4_path_obj.parent.name
    folder = mp4_path_obj.parent.parent.name
    clip_id = mp4_path_obj.parent.parent.parent.name
    clip_type = "im" if folder == "important-moments" else "nim"
    frame_dir = Path(frames_root) / clip_id / clip_type / group_idx / clip_name
    return str(frame_dir), clip_id, clip_type, group_idx, clip_name


def build_clean_record(
    *,
    task: str,
    row: Dict[str, str],
    frames_root: str,
    output_dir: str,
    similarity_threshold: float,
    n_frames: int,
    prefer_mp4: bool,
) -> Optional[Dict[str, object]]:
    """
    Build the clean, task-specific record used by all counterfactual modes.
    """
    mp4_path = resolve_path(row["mp4_path"])
    transcript_data = load_transcript_data(mp4_path)
    if transcript_data is None:
        return None

    similarity = float(transcript_data.get("similarity", 0.0))
    if similarity < similarity_threshold:
        return None

    local_text = str(transcript_data.get("local", "")).strip()
    global_text = str(transcript_data.get("global", "")).strip()
    if not local_text:
        return None

    frame_dir, clip_id, clip_type, group_idx, clip_name = build_frame_root(
        resolve_path(frames_root), mp4_path
    )
    frame_images, frame_source, frame_paths_abs = load_clip_visual_inputs(
        frame_dir=frame_dir,
        mp4_path=mp4_path,
        n_frames=n_frames,
        prefer_mp4=prefer_mp4,
    )
    composite_path_abs = build_clean_composite_path(
        task=task,
        clip_id=clip_id,
        clip_type=clip_type,
        group_idx=group_idx,
        clip_name=clip_name,
        frame_paths=frame_paths_abs if frame_paths_abs else None,
        frame_images=frame_images,
        output_dir=output_dir,
        write_image=True,
    )

    label = row["label"].strip().lower()
    event_type = row["event_type"].strip()
    answer = get_answer(task, label, event_type)
    if task == "event_type":
        label = answer
    prompt = build_prompt(local_text, task)
    sample_id = f"{clip_id}__{group_idx}__{clip_name}"

    return {
        "clip_id": clip_id,
        "clip_type": clip_type,
        "group_idx": group_idx,
        "clip_name": clip_name,
        "event_type": event_type,
        "label": label,
        "similarity": f"{similarity:.6f}",
        "local_text": local_text,
        "global_text": global_text,
        "prompt": prompt,
        "image_paths": to_repo_relative_path(composite_path_abs),
        "answer": answer,
        "_frame_images": frame_images,
        "_frame_source": frame_source,
        "_clean_composite_path": composite_path_abs,
        "_sample_id": sample_id,
    }


def maybe_absolute(path: str, base_dir: str) -> str:
    if not path:
        return path
    if os.path.isabs(path):
        return path
    return str(Path(base_dir) / path)


def to_repo_relative_path(path: str) -> str:
    """
    Store paths relative to the repository root whenever possible.

    This keeps the CSV portable across local and Snellius checkouts. If the
    path does not live under the repo root, fall back to stripping a leading
    slash so manually normalized paths like `/thesis_project/...` still become
    portable relative paths.
    """
    if not path:
        return path

    path_obj = Path(path)
    if not path_obj.is_absolute():
        return str(path_obj)

    try:
        return str(path_obj.resolve().relative_to(REPO_ROOT))
    except Exception:
        return path.lstrip(os.sep)


def build_sample_record(
    *,
    task: str,
    row: Dict[str, str],
    moments_root: str,
    frames_root: str,
    output_dir: str,
    n_frames: int,
    noise_sigma: float,
    write_noisy_frames: bool,
    noisy_frames_root: str,
    prefer_mp4: bool,
    language_perturbation_fraction: float,
    mode: str,
    similarity_threshold: float,
) -> Optional[Dict[str, str]]:
    clean_record = build_clean_record(
        task=task,
        row=row,
        frames_root=frames_root,
        output_dir=output_dir,
        similarity_threshold=similarity_threshold,
        n_frames=n_frames,
        prefer_mp4=prefer_mp4,
    )
    if clean_record is None:
        return None

    clip_id = str(clean_record["clip_id"])
    clip_type = str(clean_record["clip_type"])
    group_idx = str(clean_record["group_idx"])
    clip_name = str(clean_record["clip_name"])
    label = str(clean_record["label"])
    event_type = str(clean_record["event_type"])
    similarity = str(clean_record["similarity"])
    local_text = str(clean_record["local_text"])
    global_text = str(clean_record["global_text"])
    prompt = str(clean_record["prompt"])
    answer = str(clean_record["answer"])
    frame_images = list(clean_record["_frame_images"])  # type: ignore[index]
    sample_id = str(clean_record["_sample_id"])
    clean_composite_path = str(clean_record["_clean_composite_path"])
    cf_answer = get_cf_answer(task, answer, sample_id, mode)
    cf_prompt = ""
    cf_image_paths = ""
    cf_prompt_state = ""
    cf_prompt_changes = ""

    if mode in {"language_only", "both"}:
        cf_text, cf_prompt_state, cf_prompt_changes = build_language_counterfactual_text(
            local_text,
            global_text,
            perturbation_fraction=language_perturbation_fraction,
        )
        if mode == "both":
            cf_prompt_state = f"{cf_prompt_state}+vision_noise"
        cf_prompt = build_prompt(cf_text, task)
    elif mode == "vision_only":
        cf_prompt = prompt
        cf_prompt_state = "vision_only"
    elif mode == "random_pair":
        cf_prompt = ""
        cf_prompt_state = "random_pair"
        cf_prompt_changes = ""
    else:
        raise ValueError(f"Unknown MOMENTS mode: {mode}")

    if mode in {"vision_only", "both"}:
        noisy_path = build_shared_vision_noisy_composite_path(
            task=task,
            clip_id=clip_id,
            clip_type=clip_type,
            group_idx=group_idx,
            clip_name=clip_name,
            frame_images=frame_images,
            output_dir=output_dir,
            noise_sigma=noise_sigma,
            write_noisy_frames=write_noisy_frames,
            noisy_frames_root=noisy_frames_root,
        )
        cf_image_paths = to_repo_relative_path(noisy_path)
    elif mode == "language_only":
        cf_image_paths = to_repo_relative_path(clean_composite_path)
    elif mode == "random_pair":
        cf_image_paths = ""

    return {
        "clip_id": clip_id,
        "group_idx": group_idx,
        "clip_name": clip_name,
        "event_type": event_type,
        "label": label,
        "similarity": similarity,
        "local_text": local_text,
        "global_text": global_text,
        "prompt": prompt,
        "image_paths": to_repo_relative_path(clean_composite_path),
        "answer": answer,
        "cf_mode": mode,
        "cf_prompt": cf_prompt,
        "cf_image_paths": cf_image_paths,
        "cf_answer": cf_answer if mode != "random_pair" else "",
        "cf_prompt_state": cf_prompt_state,
        "cf_prompt_changes": cf_prompt_changes,
    }


def build_random_pair_record(
    *,
    task: str,
    clean_record: Dict[str, object],
    paired_record: Dict[str, object],
) -> Dict[str, str]:
    answer = str(clean_record["answer"])
    cf_answer = str(paired_record["answer"])
    if answer == cf_answer:
        raise ValueError("Random pair must use a sample with a different answer")

    return {
        "clip_id": str(clean_record["clip_id"]),
        "group_idx": str(clean_record["group_idx"]),
        "clip_name": str(clean_record["clip_name"]),
        "event_type": str(clean_record["event_type"]),
        "label": str(clean_record["label"]),
        "similarity": str(clean_record["similarity"]),
        "local_text": str(clean_record["local_text"]),
        "global_text": str(clean_record["global_text"]),
        "prompt": str(clean_record["prompt"]),
        "image_paths": str(clean_record["image_paths"]),
        "answer": answer,
        "cf_mode": "random_pair",
        "cf_prompt": str(paired_record["prompt"]),
        "cf_image_paths": str(paired_record["image_paths"]),
        "cf_answer": cf_answer,
        "cf_prompt_state": "random_pair",
        "cf_prompt_changes": (
            f"paired_with:{paired_record['clip_id']}__{paired_record['group_idx']}__{paired_record['clip_name']}"
        ),
    }


def write_csv(path: str, rows: Sequence[Dict[str, str]]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


def main() -> None:
    args = parse_args()
    annotations = load_annotation_rows(args.annotations_dir, args.task)
    if args.task == "goal":
        annotations = deduplicate_goal_annotations(annotations)
    if args.limit > 0:
        annotations = annotations[: args.limit]

    task_output_dir = Path(args.output_dir) / f"moments_{args.task}"
    os.makedirs(task_output_dir, exist_ok=True)

    clean_records: List[Dict[str, object]] = []
    mode_to_rows: Dict[str, List[Dict[str, str]]] = {
        "language_only": [],
        "vision_only": [],
        "both": [],
    }

    for row in annotations:
        clean_record = build_clean_record(
            task=args.task,
            row=row,
            frames_root=args.frames_root,
            output_dir=args.output_dir,
            similarity_threshold=args.similarity_threshold,
            n_frames=args.n_frames,
            prefer_mp4=args.prefer_mp4,
        )
        if clean_record is None:
            continue
        clean_records.append(clean_record)

        for mode in mode_to_rows:
            sample = build_sample_record(
                task=args.task,
                row=row,
                moments_root=args.moments_root,
                frames_root=args.frames_root,
                output_dir=args.output_dir,
                n_frames=args.n_frames,
                noise_sigma=args.noise_sigma,
                write_noisy_frames=not args.no_write_noisy_frames,
                noisy_frames_root=args.noisy_frames_root,
                prefer_mp4=args.prefer_mp4,
                language_perturbation_fraction=args.language_perturbation_fraction,
                mode=mode,
                similarity_threshold=args.similarity_threshold,
            )
            if sample is not None:
                mode_to_rows[mode].append(sample)

    random_pair_rows: List[Dict[str, str]] = []
    answer_to_records: Dict[str, List[Dict[str, object]]] = {}
    for record in clean_records:
        answer_to_records.setdefault(str(record["answer"]), []).append(record)

    for record in clean_records:
        answer = str(record["answer"])
        other_answer = "no" if answer == "yes" else "yes"
        candidates = answer_to_records.get(other_answer, [])
        if not candidates:
            continue
        sample_id = str(record["_sample_id"])
        pair_idx = sample_seed(args.task, "random_pair", sample_id) % len(candidates)
        paired = candidates[pair_idx]
        if paired["_sample_id"] == record["_sample_id"] and len(candidates) > 1:
            pair_idx = (pair_idx + 1) % len(candidates)
            paired = candidates[pair_idx]
        if paired["_sample_id"] == record["_sample_id"]:
            continue
        random_pair_rows.append(
            build_random_pair_record(task=args.task, clean_record=record, paired_record=paired)
        )

    mode_to_rows["random_pair"] = random_pair_rows

    for mode, rows in mode_to_rows.items():
        csv_path = task_output_dir / f"{mode}_data.csv"
        write_csv(str(csv_path), rows)
        print(f"{mode}: wrote {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
