"""Encode paper text to L2-normalized float32 vectors of size TEXT_DIM."""

from __future__ import annotations

import hashlib
import json
import re

import numpy as np

from src import config
from src.schema import paper_text

LAST_ENCODER = config.HASH_ENCODER_NAME

_TOKEN = re.compile(r"[a-z0-9]+")


def _hash_embed(texts: list[str]) -> np.ndarray:
    vectors = np.zeros((len(texts), config.TEXT_DIM), dtype=np.float32)
    for row, text in enumerate(texts):
        tokens = _TOKEN.findall(text.lower()) or ["empty"]
        vec = vectors[row]
        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).digest()
            bucket = int.from_bytes(digest[:4], "little") % config.TEXT_DIM
            sign = np.float32(1.0 if digest[4] % 2 == 0 else -1.0)
            vec[bucket] += sign
        norm = float(np.linalg.norm(vec))
        vec /= np.float32(max(norm, 1e-8))
    return vectors


def _neural_embed(texts: list[str]) -> np.ndarray | None:
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(config.ENCODER_NAME)
        raw = model.encode(
            list(texts),
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        array = np.asarray(raw, dtype=np.float32)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.shape != (len(texts), config.TEXT_DIM):
            return None
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        return array / np.maximum(norms, np.float32(1e-8))
    except Exception:
        return None


def encode_texts(texts: list[str], prefer_neural: bool = True) -> np.ndarray:
    global LAST_ENCODER
    if prefer_neural:
        neural = _neural_embed(texts)
        if neural is not None:
            LAST_ENCODER = config.ENCODER_NAME
            return neural
    LAST_ENCODER = config.HASH_ENCODER_NAME
    return _hash_embed(texts)


def _read_jsonl(path) -> list[dict]:
    papers: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                papers.append(json.loads(line))
    return papers


def main() -> None:
    papers = _read_jsonl(config.ENRICHED_PATH)
    vectors = encode_texts([paper_text(paper) for paper in papers], prefer_neural=True)
    config.EMBEDDINGS_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(config.EMBEDDINGS_PATH, np.asarray(vectors, dtype=np.float32))
    meta = {"encoder": LAST_ENCODER, "n": int(vectors.shape[0])}
    config.EMBEDDING_META_PATH.write_text(json.dumps(meta) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
