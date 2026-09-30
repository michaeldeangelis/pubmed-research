"""Metascience v2 pathway configurations (EGFR/ERBB and PI3K/AKT/mTOR).

Contract: ``experiments.md`` entry "2026-09-30 metascience v2: other
pathways". Each pathway is a separate corpus built with the same code as the
v1 RAS/MAPK corpus (``src.meta.corpus``, ``src.meta.features``,
``src.meta.impact`` with ``--pathway <name>``); outputs go under
``data/meta/pathways/<name>/``. PMIDs in the v1 corpus
(``data/meta/papers.jsonl``) are dropped at corpus build time.

Each entry holds:

* ``terms``: the Europe PMC TITLE_ABS terms, exactly the ledger lists.
* ``query``: the Europe PMC query (terms OR-ed, ``PUB_YEAR:[2000 TO 2015]``,
  ``SRC:MED``), in the same shape as the v1 ingest query.
* ``gene_groups``: ``(name, pattern)`` pairs; the first match over title +
  abstract wins, ``other`` if none (the covariate levels are
  ``gene_group_levels``).
* ``on_topic``: patterns for eligibility rule (b): a case-sensitive gene
  mention or a drug name (any case).

Pattern conventions (same spirit as the v1 RAS patterns in
``src.meta.features``):

* Gene symbols are CASE-SENSITIVE. An upper-case-initial symbol must not
  follow an upper-case letter or a digit (so ``OTHER2`` is not HER2), but may
  follow a lower-case letter, because Europe PMC runs section labels into the
  text ("ResultsEGFR") and phospho prefixes are common ("pAkt", "pEGFR"). A
  lower-case-initial symbol (``erbB2``, ``mTOR``, ``pten``) must not follow
  any letter or digit.
* A trailing lower-case letter blocks a match (German "Aktuelle",
  "Aktivitat"), except the listed suffixes: plural ``s``, inhibitor ``i`` /
  ``is`` (EGFRi, mTORi), EGFR variant ``vIII`` and ``wt``/``mut``.
* A trailing digit blocks a match unless it is part of the symbol
  (``HER2`` matches, ``HER21`` does not; ``AKT`` also matches ``AKT1``-``AKT3``).

Ambiguities resolved:

* ``eGFR`` (estimated glomerular filtration rate) is very common and Europe
  PMC's TITLE_ABS search is case-insensitive, so the EGFR corpus contains many
  nephrology papers. The EGFR pattern accepts ``EGFR`` and ``Egfr`` (mouse)
  only; ``eGFR`` and all-lower-case ``egfr`` never match, so those papers fall
  out as ``off_topic``.
* ``neu`` alone is not accepted (neutrophil, neuraminidase, "neu-" prefixes).
  HER2 needs ``HER2``/``Her2``/``HER-2``/``Her-2``, ``ERBB2``/``ErbB2``/
  ``erbB2``/``Erbb2`` (with an optional hyphen before the 2) or
  ``c-erbB-2``; ``HER2/neu`` matches through its HER2 part.
* All-lower-case ``her2`` / ``her-2`` is rejected (collides with the pronoun
  in run-together or OCR-damaged text such as "her2nd"); ``Her2`` is accepted.
* ``HER1``/``ErbB1``/``ERBB1``/``c-erbB-1`` count as EGFR; ``HER3``/``ErbB3``
  count as on-topic (ledger term ERBB3) but fall in gene group ``other``.
  ERBB4/HER4 is not on the ledger list and does not count.
* ``PTEN``, ``Pten`` (mouse) and ``pten`` (zebrafish) match; ``PTENP1`` (the
  PTEN pseudogene) matches, because it is PTEN work.
* ``AKT`` and ``Akt`` (optionally 1-3, optional plural ``s``) match;
  all-lower-case ``akt`` does not, and ``Aktuelle`` / ``Aktivitat`` do not.
  ``AKTs``/``Akts`` (the isoforms, plural) are accepted: unlike the v1
  "HRAs" case there is no common non-gene acronym "AKTs".
* ``p110alpha``/``p110-alpha``/``p110α`` count as PIK3CA; bare ``p110`` does
  not (it is also p110beta/delta/gamma and other proteins).
* ``mTOR``, ``MTOR``, ``Mtor``, ``mtor`` and ``mTORC1``/``mTORC2`` match.
* Mixed-case variants seen in abstracts are accepted: ``EGFr``, ``EgfR``
  (any E + g/G f/F r/R, upper-case E), ``mTor``/``mToR``/``mTorC1``/
  ``mTORc1`` (any casing of m-TOR-C), ``pTEN``. Still rejected: ``eGFR``,
  all-lower-case ``egfr``/``her2``/``erbb2``/``akt1``.
* Two pathway-specific exclusions run after the on-topic check (review
  2026-09-29; before ``non_primary``):
  ``off_topic_plant`` (PI3K): the only on-topic match is the AKT pattern and
  the paper has a strong plant signal (plant-organism MeSH, or text
  Arabidopsis/OsAKT/AtAKT/Populus), or a weak one (MeSH Arabidopsis
  Proteins/Plant Proteins/Plant Roots/Potassium Channels etc., text
  rice/plant) with neither MeSH Humans nor Animals. Plant AKT1 is a potassium
  channel, and MEDLINE mis-indexes mammalian Akt1 papers with "Arabidopsis
  Proteins", so that heading alone is not enough. ``off_topic_kidney`` (EGFR): the only on-topic match is the EGFR
  pattern, the text uses EGFR as a filtration rate (``GFR_TEXT``) and there is
  no receptor signal (``RECEPTOR_TEXT``).
* The drug lists are exactly the ledger lists (rapamycin/sirolimus and
  PI3K-generic mentions such as "PI3K" do not count).
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PATHWAYS_DIR = ROOT / "data" / "meta" / "pathways"
YEAR_RANGE = "[2000 TO 2015]"

# Trailing context: no lower-case letter (so "Krasavin"-type words fail) and
# no digit (so HER21 is not HER2).
_END = r"(?![a-z0-9])"


def _upper(symbols: str) -> str:
    """Upper-case-initial symbol: not after an upper-case letter or digit."""
    return r"(?<![A-Z0-9])(?:" + symbols + ")"


def _lower(symbols: str) -> str:
    """Lower-case-initial symbol: not after any letter or digit."""
    return r"(?<![A-Za-z0-9])(?:" + symbols + ")"


# --- EGFR/ERBB ------------------------------------------------------------------

EGFR_PATTERN = re.compile(
    "(?:"
    + _upper(r"E[Gg][Ff][Rr]") + r"(?:vIII|wt|mut|is|s|i)?" + _END
    + "|" + _upper(r"ERBB-?1|ErbB-?1|Erbb-?1|HER-?1|Her-?1") + _END
    + "|" + _lower(r"(?:c-)?erbB-?1") + _END
    + ")"
)
ERBB2_PATTERN = re.compile(
    "(?:"
    + _upper(r"ERBB-?2|ErbB-?2|Erbb-?2|HER-?2|Her-?2") + r"(?:s)?" + _END
    + "|" + _lower(r"(?:c-)?erbB-?2") + _END
    + ")"
)
ERBB3_PATTERN = re.compile(
    "(?:"
    + _upper(r"ERBB-?3|ErbB-?3|Erbb-?3|HER-?3|Her-?3") + _END
    + "|" + _lower(r"(?:c-)?erbB-?3") + _END
    + ")"
)
EGFR_DRUG_PATTERN = re.compile(
    r"\b(erlotinib|gefitinib|osimertinib|afatinib|lapatinib|cetuximab|"
    r"panitumumab|trastuzumab)\b",
    re.I,
)

# --- PI3K/AKT/mTOR --------------------------------------------------------------

PIK3CA_PATTERN = re.compile(
    "(?:"
    + _upper(r"PIK3CA|Pik3ca") + _END
    + "|" + _lower(r"pik3ca") + _END
    + "|" + _lower(r"p110(?:-?α|-?alpha|\sα|\salpha)") + r"(?![A-Za-z0-9])"
    + ")"
)
PTEN_PATTERN = re.compile(
    "(?:"
    + _upper(r"PTEN|Pten") + _END
    + "|" + _lower(r"pten|pTEN") + r"(?:[ab])?" + _END
    + ")"
)
AKT_PATTERN = re.compile(_upper(r"AKT|Akt") + r"[1-3]?s?" + _END)
MTOR_PATTERN = re.compile(
    "(?:"
    + _upper(r"M[Tt][Oo][Rr]") + r"(?:[Cc][12]?)?(?:is|s|i)?" + _END
    + "|" + _lower(r"m[Tt][Oo][Rr]") + r"(?:[Cc][12]?)?(?:is|s|i)?" + _END
    + ")"
)
PI3K_DRUG_PATTERN = re.compile(
    r"\b(everolimus|temsirolimus|alpelisib|idelalisib)\b", re.I
)


# --- pathway-specific exclusions (applied after the on-topic check) -------------

_MARKUP = re.compile(r"<[^>]+>")


def _text(paper: dict) -> str:
    return _MARKUP.sub("", f"{paper.get('title') or ''} {paper.get('abstract') or ''}")


def _mesh(paper: dict) -> set[str]:
    out = set()
    for m in paper.get("mesh") or []:
        t = re.sub(r"\s+", " ", str(m).split("/", 1)[0].replace("*", " ")).strip().casefold()
        if t:
            out.add(t)
    return out


# Plant AKT1 (Arabidopsis/rice potassium channel AKT1, "OsAKT1"): a PI3K paper
# whose only on-topic match is the AKT pattern and that shows plant signals.
# Strong signals exclude on their own: a plant-organism MeSH heading or the
# text Arabidopsis / OsAKT / AtAKT / Populus. Weak signals exclude only when
# the paper has neither MeSH "Humans" nor "Animals": MEDLINE's automatic
# mapping indexes many mammalian Akt1 papers with "Arabidopsis Proteins" /
# "Plant Proteins" (the AKT1 supplementary concept is the Arabidopsis
# channel), and mammalian natural-product papers mention "plant" or "rice".
PLANT_ORGANISM_MESH = frozenset(t.casefold() for t in (
    "Plants", "Plants, Genetically Modified", "Arabidopsis", "Oryza", "Populus",
    "Nicotiana", "Hordeum", "Triticum", "Seedlings",
))
# "Zea mays" is weak: MeSH also uses it for corn-derived food (corn syrup).
PLANT_WEAK_MESH = frozenset(t.casefold() for t in (
    "Arabidopsis Proteins", "Plant Proteins", "Plant Roots", "Plant Leaves",
    "Plant Shoots", "Potassium Channels", "Zea mays",
))
PLANT_STRONG_TEXT = re.compile(r"\b(?:Arabidopsis|OsAKT\d?|AtAKT\d?|Populus)\b", re.I)
PLANT_WEAK_TEXT = re.compile(r"\b(?:rice|plants?)\b", re.I)
_HUMAN_OR_ANIMAL = frozenset({"humans", "animals"})


def plant_akt_exclusion(paper: dict) -> str | None:
    text = _text(paper)
    others = (PIK3CA_PATTERN, PTEN_PATTERN, MTOR_PATTERN, PI3K_DRUG_PATTERN)
    if not AKT_PATTERN.search(text) or any(p.search(text) for p in others):
        return None
    mesh = _mesh(paper)
    if mesh & PLANT_ORGANISM_MESH or PLANT_STRONG_TEXT.search(text):
        return "off_topic_plant"
    weak = bool(mesh & PLANT_WEAK_MESH) or bool(PLANT_WEAK_TEXT.search(text))
    if weak and not mesh & _HUMAN_OR_ANIMAL:
        return "off_topic_plant"
    return None


# Kidney "EGFR" (estimated glomerular filtration rate written in capitals): an
# EGFR paper whose only on-topic match is the EGFR pattern, whose text uses
# EGFR as a filtration rate, and that has no EGFR-receptor signal.
GFR_TEXT = re.compile(
    r"glomerular filtration|filtration rate|m[lL]\s*/\s*min|"
    r"E[Gg][Ff][Rr]\s*(?:of|<|>|=|\u2264|\u2265|<=|>=)\s*\d",
    re.I,
)
RECEPTOR_TEXT = re.compile(
    r"receptors?|tyrosine[- ]kinases?|\bTKIs?\b|mutat\w*|mutant|"
    r"epidermal growth factor|\bEGF\b|\bHER\b|ErbB|erbB|ERBB|"
    r"erlotinib|gefitinib|osimertinib|afatinib|lapatinib|cetuximab|panitumumab|trastuzumab",
    re.I,
)


def kidney_egfr_exclusion(paper: dict) -> str | None:
    text = _text(paper)
    others = (ERBB2_PATTERN, ERBB3_PATTERN, EGFR_DRUG_PATTERN)
    if not EGFR_PATTERN.search(text) or any(p.search(text) for p in others):
        return None
    if GFR_TEXT.search(text) and not RECEPTOR_TEXT.search(text):
        return "off_topic_kidney"
    return None


def _query(terms: tuple[str, ...]) -> str:
    joined = " OR ".join(f'TITLE_ABS:"{t}"' for t in terms)
    return f"({joined}) AND PUB_YEAR:{YEAR_RANGE} AND SRC:MED"


EGFR_TERMS = (
    "EGFR", "ERBB2", "HER2", "ERBB3", "erlotinib", "gefitinib", "osimertinib",
    "afatinib", "lapatinib", "cetuximab", "panitumumab", "trastuzumab",
)
PI3K_TERMS = (
    "PIK3CA", "PTEN", "AKT1", "MTOR", "mTORC1", "everolimus", "temsirolimus",
    "alpelisib", "idelalisib",
)

PATHWAYS: dict[str, dict] = {
    "egfr": {
        "label": "EGFR/ERBB",
        "terms": EGFR_TERMS,
        "query": _query(EGFR_TERMS),
        "gene_groups": (("EGFR", EGFR_PATTERN), ("ERBB2", ERBB2_PATTERN)),
        "gene_group_levels": ("EGFR", "ERBB2", "other"),
        "on_topic": (EGFR_PATTERN, ERBB2_PATTERN, ERBB3_PATTERN, EGFR_DRUG_PATTERN),
        "extra_exclusion": kidney_egfr_exclusion,
        "extra_reasons": ("off_topic_kidney",),
    },
    "pi3k": {
        "label": "PI3K/AKT/mTOR",
        "terms": PI3K_TERMS,
        "query": _query(PI3K_TERMS),
        "gene_groups": (
            ("PIK3CA", PIK3CA_PATTERN),
            ("PTEN", PTEN_PATTERN),
            ("AKT", AKT_PATTERN),
            ("MTOR", MTOR_PATTERN),
        ),
        "gene_group_levels": ("PIK3CA", "PTEN", "AKT", "MTOR", "other"),
        "on_topic": (PIK3CA_PATTERN, PTEN_PATTERN, AKT_PATTERN, MTOR_PATTERN, PI3K_DRUG_PATTERN),
        "extra_exclusion": plant_akt_exclusion,
        "extra_reasons": ("off_topic_plant",),
    },
}
NAMES = tuple(PATHWAYS)


def get(name: str) -> dict:
    if name not in PATHWAYS:
        raise KeyError(f"unknown pathway {name!r}; choose from {', '.join(NAMES)}")
    return PATHWAYS[name]


def pathway_dir(name: str, base: Path = PATHWAYS_DIR) -> Path:
    get(name)
    return base / name
