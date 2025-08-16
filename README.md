# Spectral Clustering (CE282 — Final Project)

**Notebook:** `Spectral-Clustering.ipynb`
**Student:** Mohammad Sina Beyrami — 400105433

## Overview

This project implements **spectral clustering from scratch** on:

1. hand-constructed graphs, and
2. 2-D concentric circles generated via `sklearn.datasets.make_circles`.

Pipeline (unnormalized Laplacian):

1. Build a similarity graph $G=(V,E)$
2. Construct **adjacency** $A$ and **degree** $D$
3. Compute **Laplacian** $L = D - A$
4. Eigendecompose $L$ → eigenvalues/eigenvectors
5. Use the **Fiedler vector** (second eigenvector) or the first $k$ non-trivial eigenvectors as an embedding
6. **K-Means** in the embedded space to assign cluster labels

The notebook also visualizes how the Laplacian **eigenvalues** evolve as new edges are added, and then applies the same spectral recipe to a **k-NN graph** over the circles to separate non-linearly separable clusters.

## What’s inside

* **Graph class**: minimal undirected graph with `add_node`, `connect`, `disconnect`
* **Matrices**: adjacency $A$, degree $D$, Laplacian $L$
* **Spectrum**: sorted eigenvalues/eigenvectors (`numpy.linalg.eigh`)
* **Clustering**: K-Means on Laplacian eigenvectors
* **Datasets**: `make_circles` + kNN graph (`kneighbors_graph`)
* **Plots**: eigenvalue progression, clustered points

## Requirements

* Python ≥ 3.9
* `numpy`, `matplotlib`, `scikit-learn`
* (Optional) `graphviz` and `python-graphviz` for colored subgraph rendering

Install (example):

```bash
pip install numpy matplotlib scikit-learn graphviz
# For graph rendering:
pip install python-graphviz
```

## How to run

1. Open the notebook and run cells top-to-bottom.
2. First section: build a small graph, add edges stepwise, and **plot eigenvalues** to see connectivity effects (zero eigenvalue multiplicity ↔ number of components; Fiedler value ↔ algebraic connectivity).
3. Second section: generate **two circles**, build a **k-NN graph**, compute Laplacian spectrum, and split using the **sign of the Fiedler vector** (or K-Means on the first two non-trivial eigenvectors).

## Notes on Interpretation

* **Non-negativity**: Laplacian eigenvalues are ≥ 0.
* **Zero eigenvalues**: count equals the number of **connected components**.
* **Fiedler value**: the **second smallest** eigenvalue increases as the graph becomes better connected; its eigenvector often cleanly separates weakly connected parts.

## Known pitfalls & minimal fixes

A few small issues can cause runtime errors; here are concise corrections:

1. **Node indexing mismatch (0/1-based)**
   Your `Graph.add_node()` creates nodes as `{1, 2, …, N}`, but you connect with `connect(0, 1)` etc. Use **0-based** consistently:

```python
class Graph:
    def __init__(self):
        self.N = 0
        self.nodes = set()
        self.vertices = {}

    def add_node(self):
        self.nodes.add(self.N)
        self.vertices[self.N] = set()
        self.N += 1

    def connect(self, A, B):
        assert 0 <= A < self.N and 0 <= B < self.N and A != B
        self.vertices[A].add(B)
        self.vertices[B].add(A)
```

2. **Function name shadowing** (UnboundLocalError risk)
   Rename helper functions to avoid reusing names as local variables:

```python
def get_adjacency_matrix(graph):
    A = np.zeros((graph.N, graph.N), dtype=int)
    for u in graph.nodes:
        for v in graph.vertices[u]:
            A[u, v] = 1
            A[v, u] = 1
    return A

def get_degree_matrix(graph):
    D = np.zeros((graph.N, graph.N), dtype=int)
    for u in graph.nodes:
        D[u, u] = len(graph.vertices[u])
    return D

def get_laplacian_matrix(graph):
    A = get_adjacency_matrix(graph)
    D = get_degree_matrix(graph)
    return D - A
```

3. **Eigen helper signature & usage**
   Make `K` optional and always return sorted spectrum:

```python
def laplacian_spectrum(graph, k=None):
    L = get_laplacian_matrix(graph)
    vals, vecs = np.linalg.eigh(L)
    if k is None:
        # return full spectrum
        return vals, vecs
    # skip the trivial eigenpair at index 0
    return vals[1:k+1], vecs[:, 1:k+1]
```

Then call `laplacian_spectrum(g)` for plotting all eigenvalues, or `laplacian_spectrum(g, k=2)` for clustering.

4. **k-NN graph symmetry (circles)**
   `kneighbors_graph` can be asymmetric; symmetrize before building $L$:

```python
from sklearn.neighbors import kneighbors_graph
A = kneighbors_graph(X, n_neighbors=5, mode='connectivity', include_self=False).toarray()
A = np.maximum(A, A.T)  # symmetrize
D = np.diag(A.sum(1))
L = D - A
vals, vecs = np.linalg.eigh(L)
# Use Fiedler vector for 2 clusters:
clusters = vecs[:, 1] > 0
```

5. **K-Means on eigenvectors**
   For $k$ clusters, fit on the first $k$ non-trivial eigenvectors:

```python
from sklearn.cluster import KMeans
k = 3
_, U = laplacian_spectrum(graph, k=k)
labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(U)
```

## Results

* **Toy graph**: as edges are added, the **second eigenvalue** grows (better connectivity); zero eigenvalue stays single once the graph is connected.
* **Two circles**: the sign of the **Fiedler vector** cleanly separates inner/outer rings; K-Means on the first two non-trivial eigenvectors yields the same split.

## References

* Luxburg, U. von (2007). *A Tutorial on Spectral Clustering*.
* Shi & Malik (2000). *Normalized Cuts and Image Segmentation*.

---

## License:

MIT © Sina Beyrami

