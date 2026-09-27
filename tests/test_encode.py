import numpy as np

from src.config import TEXT_DIM
from src.encode.encoder import encode_texts


def test_hash_encoder_is_deterministic_768():
    texts = ["KRAS G12C inhibitor in NSCLC", "germline BRAF variant"]
    first = encode_texts(texts, prefer_neural=False)
    second = encode_texts(texts, prefer_neural=False)
    assert first.shape == (2, TEXT_DIM)
    assert first.dtype == np.float32
    assert np.allclose(first, second)
    norms = np.linalg.norm(first, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-4)
    assert not np.allclose(first[0], first[1])
