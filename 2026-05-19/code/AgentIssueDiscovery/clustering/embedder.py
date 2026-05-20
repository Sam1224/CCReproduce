"""Text embedding for clustering recalled video candidates.

Paper §3.2: Videos recalled by the agent are embedded using a multimodal encoder.
Here we use a toy TF-IDF + SVD embedding for demonstration.
In production: use a real video/text encoder (CLIP, InternVideo, etc.)
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize


class VideoEmbedder:
    """Embed video descriptions for clustering.

    Paper uses multimodal embeddings from video frames + text.
    This toy implementation uses TF-IDF on text descriptions.
    """

    def __init__(self, n_components: int = 64):
        self.n_components = n_components
        self.vectorizer = TfidfVectorizer(
            max_features=2000,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self.svd = TruncatedSVD(n_components=n_components, random_state=42)
        self._fitted = False

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        """Fit on texts and return embeddings."""
        tfidf = self.vectorizer.fit_transform(texts)
        embeddings = self.svd.fit_transform(tfidf)
        self._fitted = True
        return normalize(embeddings, norm="l2")

    def transform(self, texts: list[str]) -> np.ndarray:
        """Transform texts using fitted embedder."""
        if not self._fitted:
            raise RuntimeError("Call fit_transform first.")
        tfidf = self.vectorizer.transform(texts)
        embeddings = self.svd.transform(tfidf)
        return normalize(embeddings, norm="l2")
