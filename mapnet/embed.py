"""
Dense text embeddings, used by the embedding-search baseline (and available as
an upgrade path for the router).

- OllamaEmbedder: a local embedding model through Ollama (default nomic-embed-text).
  Vectors are cached on disk by text hash, so re-runs are fast.
- LsaEmbedder: no model at all (TF-IDF + SVD), so the harness still runs where
  Ollama isn't installed, e.g. in CI. Much weaker than a real embedding model;
  results with it say nothing about embedding search.
"""
from __future__ import annotations

import hashlib
import json
import os
import urllib.request
from pathlib import Path

import numpy as np


def _normalize(v):
    n = np.linalg.norm(v, axis=1, keepdims=True)
    n[n == 0] = 1
    return v / n


class OllamaEmbedder:
    def __init__(self, model=None, host=None, cache_dir=".cache/embeddings", batch=32):
        self.model = model or os.environ.get("MAPNET_EMBED_MODEL", "nomic-embed-text")
        self.host = (host or os.environ.get("OLLAMA_HOST", "http://localhost:11434")).rstrip("/")
        if not self.host.startswith("http"):
            self.host = "http://" + self.host
        self.batch = batch
        self.cache_path = Path(cache_dir) / f"{self.model.replace(':', '_').replace('/', '_')}.npz"
        self._cache = {}
        if self.cache_path.exists():
            d = np.load(self.cache_path)
            self._cache = dict(zip(d["keys"].tolist(), d["vecs"]))
        self.name = f"{self.model} (Ollama)"

    def available(self):
        try:
            with urllib.request.urlopen(f"{self.host}/api/tags", timeout=2) as r:
                names = [m["name"] for m in json.load(r).get("models", [])]
            return any(n == self.model or n.split(":")[0] == self.model for n in names)
        except Exception:
            return False

    def _call(self, texts, prefix):
        body = json.dumps({"model": self.model, "input": [prefix + t for t in texts]}).encode()
        req = urllib.request.Request(f"{self.host}/api/embed", body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=600) as r:
            return np.array(json.load(r)["embeddings"], dtype=np.float32)

    def embed(self, texts, kind="document", progress=None):
        # nomic-style task prefixes; harmless for models that don't use them
        prefix = "search_query: " if kind == "query" else "search_document: "
        keys = [hashlib.sha1((prefix + t).encode()).hexdigest() for t in texts]
        todo = [i for i, k in enumerate(keys) if k not in self._cache]
        for s in range(0, len(todo), self.batch):
            idx = todo[s:s + self.batch]
            vecs = self._call([texts[i][:8000] for i in idx], prefix)
            for i, v in zip(idx, vecs):
                self._cache[keys[i]] = v
            if progress:
                progress(min(s + self.batch, len(todo)), len(todo))
        if todo and kind == "document":
            self.save()
        return _normalize(np.stack([self._cache[k] for k in keys])) if texts else np.zeros((0, 1))

    def save(self):
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self._cache)
        np.savez(self.cache_path, keys=np.array(keys), vecs=np.stack([self._cache[k] for k in keys]))


class LsaEmbedder:
    """Fit once on the corpus; new texts are projected with the same model."""

    name = "LSA (no model; weak stand-in)"

    def __init__(self, dims=256):
        self.dims = dims
        self._fitted = None

    def available(self):
        return True

    def fit(self, texts):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer
        tf = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=2, max_features=100_000)
        X = tf.fit_transform(texts)
        svd = TruncatedSVD(n_components=min(self.dims, X.shape[1] - 1), random_state=0).fit(X)
        self._fitted = (tf, svd)

    def embed(self, texts, kind="document", progress=None):
        if self._fitted is None:
            self.fit(texts)
        tf, svd = self._fitted
        return _normalize(svd.transform(tf.transform(texts)).astype(np.float32))
