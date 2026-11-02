"""Graph-to-table features, written from the published method description.

Each node gets up to four blocks of columns:
  raw         its own attributes
  structure   degree and PageRank
  position    Laplacian eigenvectors with the smallest non-zero eigenvalues
  neighbours  averaged attributes of its neighbours, or Beta wavelet filter outputs

No labels are used, so the same table works for any choice of labelled nodes.
"""
import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from scipy.special import comb

ALL_BLOCKS = ("raw", "structure", "position", "neighbours")


def clean_adjacency(adj):
    """Return an undirected, unweighted CSR matrix with no self loops."""
    adj = sp.csr_matrix(adj, dtype=np.float64)
    adj = adj + adj.T
    adj.setdiag(0)
    adj.eliminate_zeros()
    adj.data[:] = 1.0
    return adj


def normalized_laplacian(adj):
    """Return L = I - D^-1/2 A D^-1/2 for a symmetric adjacency matrix."""
    degree = np.asarray(adj.sum(axis=1)).ravel()
    with np.errstate(divide="ignore"):
        inv_sqrt = np.where(degree > 0, degree**-0.5, 0.0)
    scale = sp.diags(inv_sqrt)
    return sp.identity(adj.shape[0], format="csr") - scale @ adj @ scale


def pagerank(adj, damping=0.85, iterations=100, tol=1e-8):
    """Power iteration on the column-stochastic transition matrix."""
    n = adj.shape[0]
    out_degree = np.asarray(adj.sum(axis=1)).ravel()
    inv = np.where(out_degree > 0, 1.0 / np.maximum(out_degree, 1e-12), 0.0)
    transition = adj.T @ sp.diags(inv)
    rank = np.full(n, 1.0 / n)
    for _ in range(iterations):
        dangling = rank[out_degree == 0].sum() / n
        new = damping * (transition @ rank + dangling) + (1 - damping) / n
        done = np.abs(new - rank).sum() < tol
        rank = new
        if done:
            break
    return rank


def laplacian_positions(laplacian, k=16, spare=8, seed=0):
    """Eigenvectors for the k smallest non-zero eigenvalues of L.

    Lanczos on 2I - L finds them without factorising L, which keeps memory low on dense graphs.
    Each connected component adds one zero eigenvalue, so a few spare vectors are computed.
    """
    n = laplacian.shape[0]
    flipped = 2 * sp.identity(n, format="csr") - laplacian
    start = np.random.default_rng(seed).normal(size=n)
    values, vectors = eigsh(flipped.astype(np.float64), k=k + spare, which="LA", v0=start)
    eigenvalues = 2 - values
    order = np.argsort(eigenvalues)
    nonzero = [i for i in order if eigenvalues[i] > 1e-6][:k]
    return vectors[:, nonzero]


def neighbour_means(adj, x, layers=2):
    """Mean of neighbours' attributes, repeated for each hop. Isolated nodes get zeros."""
    degree = np.asarray(adj.sum(axis=1)).ravel()
    walk = sp.diags(np.where(degree > 0, 1.0 / np.maximum(degree, 1e-12), 0.0)) @ adj
    outputs, current = [], x
    for _ in range(layers):
        current = walk @ current
        outputs.append(current)
    return np.hstack(outputs)


def beta_filter_coefficients(p, q):
    """Power-series coefficients in L of the Beta(p+1, q+1) wavelet on x = L / 2."""
    norm = (p + q + 1) * comb(p + q, p)
    coefficients = np.zeros(p + q + 1)
    for j in range(q + 1):
        coefficients[p + j] = norm * comb(q, j) * (-1) ** j / 2 ** (p + j + 1)
    return coefficients


def beta_wavelets(laplacian, x, order=2):
    """Apply all order + 1 Beta wavelet filters with p + q = order, and join the columns."""
    powers = [x]
    for _ in range(order):
        powers.append(laplacian @ powers[-1])
    outputs = []
    for p in range(order + 1):
        coefficients = beta_filter_coefficients(p, order - p)
        outputs.append(sum(c * power for c, power in zip(coefficients, powers)))
    return np.hstack(outputs)


def flatten_graph(adj, x, blocks=ALL_BLOCKS, neighbour_type="mean", layers=2, k=16, seed=0):
    """Return (table, column_names) for a scipy sparse graph and an attribute matrix."""
    unknown = set(blocks) - set(ALL_BLOCKS)
    if unknown:
        raise ValueError(f"Unknown blocks: {sorted(unknown)}")
    adj = clean_adjacency(adj)
    columns, names = [], []

    def add(values, prefix):
        columns.append(values)
        names.extend(f"{prefix}_{i}" for i in range(values.shape[1]))

    if "raw" in blocks:
        add(x, "raw")
    if "structure" in blocks:
        degree = np.asarray(adj.sum(axis=1)).ravel()
        add(np.column_stack([degree, pagerank(adj)]), "structure")
    needs_laplacian = "position" in blocks or ("neighbours" in blocks and neighbour_type == "wavelet")
    laplacian = normalized_laplacian(adj) if needs_laplacian else None
    if "position" in blocks:
        add(laplacian_positions(laplacian, k=k, seed=seed), "position")
    if "neighbours" in blocks:
        if neighbour_type == "mean":
            add(neighbour_means(adj, x, layers=layers), "neighbour")
        elif neighbour_type == "wavelet":
            add(beta_wavelets(laplacian, x, order=layers), "wavelet")
        else:
            raise ValueError(f"Unknown neighbour_type: {neighbour_type}")
    return np.hstack(columns), names
