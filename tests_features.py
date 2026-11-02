"""Fast checks that need no dataset. Run: python tests_features.py"""
import numpy as np
import scipy.sparse as sp
from graph2table.evaluate import scores, summarize
from graph2table.features import ALL_BLOCKS, beta_filter_coefficients, clean_adjacency, flatten_graph, normalized_laplacian, pagerank

n = 200
rng = np.random.default_rng(0)
rows, cols = np.triu_indices(n, 1)
keep = rng.random(rows.size) < 0.08
a = sp.coo_matrix((np.ones(keep.sum()), (rows[keep], cols[keep])), shape=(n, n)).tocsr()
a = a + a.T
x = rng.normal(size=(n, 5))

lap = normalized_laplacian(a)
assert abs(lap - lap.T).max() < 1e-10, "Laplacian must be symmetric"
pr = pagerank(a)
assert abs(pr.sum() - 1) < 1e-6, "PageRank must sum to 1"
# Beta(p+1, q+1) density on [0, 1] integrates to 1. Filter g(L) = beta(L/2) / 2 over L in [0, 2].
for p, q in [(0, 2), (1, 1), (2, 0)]:
    c = beta_filter_coefficients(p, q)
    grid = np.linspace(0, 2, 20001)
    values = sum(ci * grid**i for i, ci in enumerate(c))
    assert abs(values.mean() * 2 - 1) < 1e-3, f"Filter ({p},{q}) must integrate to 1 over [0, 2]"
    assert values.min() > -1e-9, "Filter must be non-negative on the spectrum"
table, names = flatten_graph(a, x, k=8, neighbour_type="wavelet", layers=2)
assert table.shape == (n, 5 + 2 + 8 + 15) and len(names) == table.shape[1], table.shape
assert np.isfinite(table).all()
y = np.zeros(10); y[:2] = 1
assert scores(y, np.arange(10)[::-1].astype(float))["RecK"] == 100
assert summarize([{"AUROC": 1.0}, {"AUROC": 3.0}])["AUROC"] == [2.0, 1.0]
print("all checks passed")
table2, names2 = flatten_graph(a, x, k=8, neighbour_type="mean", layers=2)
assert table2.shape == (n, 5 + 2 + 8 + 10) and len(names2) == table2.shape[1], table2.shape
only_raw, _ = flatten_graph(a, x, blocks=("raw",))
assert np.array_equal(only_raw, x)
print("all block options passed")
