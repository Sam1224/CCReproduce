import numpy as np


def kmeans(x: np.ndarray, k: int, seed: int = 7, iters: int = 30):
    """A tiny k-means implementation (numpy-only)."""

    rng = np.random.default_rng(seed)
    n = x.shape[0]

    centroids = x[rng.choice(n, size=k, replace=False)].copy()

    for _ in range(iters):
        # assign
        sim = x @ centroids.T
        labels = sim.argmax(axis=1)

        # update
        new_centroids = np.zeros_like(centroids)
        for i in range(k):
            members = x[labels == i]
            if len(members) == 0:
                new_centroids[i] = x[rng.integers(0, n)]
            else:
                c = members.mean(axis=0)
                c = c / (np.linalg.norm(c) + 1e-12)
                new_centroids[i] = c

        if np.allclose(new_centroids, centroids, atol=1e-4):
            centroids = new_centroids
            break

        centroids = new_centroids

    return labels, centroids
