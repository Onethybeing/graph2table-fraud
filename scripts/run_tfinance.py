"""T-Finance: 100 labelled nodes (20 fraud, 80 normal), 10 published splits, test on the rest.

Usage: python scripts/run_tfinance.py --graph path/to/tfinance --out results/tfinance.json
The graph file is the DGL one from the GADBench download.
"""
import argparse
import json
import pathlib
import sys
import time

import numpy as np
import scipy.sparse as sp
import xgboost as xgb

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from graph2table.evaluate import scores, summarize  # noqa: E402
from graph2table.features import flatten_graph  # noqa: E402


def load_graph(path):
    import dgl

    graph = dgl.load_graphs(str(path))[0][0]
    src, dst = graph.edges()
    n = graph.num_nodes()
    adj = sp.coo_matrix((np.ones(len(src)), (src.numpy(), dst.numpy())), shape=(n, n)).tocsr()
    x = graph.ndata["feature"].numpy().astype(np.float64)
    y = graph.ndata["label"].numpy().ravel().astype(int)
    train = graph.ndata["train_masks"].numpy()[:, 10:].astype(bool)
    test = graph.ndata["test_masks"].numpy()[:, 10:].astype(bool)
    return adj, x, y, train, test


def fit_score(table, y, train, test, seed):
    model = xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.1, max_depth=3, subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", random_state=seed, n_jobs=4,
    )
    model.fit(table[train], y[train])
    return scores(y[test], model.predict_proba(table[test])[:, 1])


VARIANTS = {
    "raw": dict(blocks=("raw",)),
    "raw+structure": dict(blocks=("raw", "structure")),
    "raw+position": dict(blocks=("raw", "position")),
    "raw+neighbours_mean": dict(blocks=("raw", "neighbours")),
    "all_mean": dict(blocks=("raw", "structure", "position", "neighbours")),
    "all_wavelet": dict(blocks=("raw", "structure", "position", "neighbours"), neighbour_type="wavelet"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    adj, x, y, train, test = load_graph(args.graph)
    print("nodes", len(y), "edges", adj.nnz, "fraud rate", round(float(y.mean()), 4), flush=True)
    print("first split: train", int(train[:, 0].sum()), "fraud", int(y[train[:, 0]].sum()), "test", int(test[:, 0].sum()), flush=True)
    results = {"base_rate_test_split0": float(y[test[:, 0]].mean())}
    for name, options in VARIANTS.items():
        start = time.time()
        table, _ = flatten_graph(adj, x, **options)
        built = time.time() - start
        rows = [fit_score(table, y, train[:, i], test[:, i], i) for i in range(train.shape[1])]
        results[name] = summarize(rows)
        results[name].update(columns=table.shape[1], feature_seconds=round(built, 1), splits=len(rows))
        print(name, results[name], flush=True)
        json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
