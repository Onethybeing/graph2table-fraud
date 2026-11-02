"""Elliptic Bitcoin data: compare a split by time with a random split, on the same labelled nodes.

Usage: python scripts/run_elliptic.py --data path/to/elliptic_bitcoin_dataset --out results/elliptic.json

Train on labelled transactions up to time step 34 and test on steps 35 to 49. Fraud patterns change
over those steps, so this is the honest setting. The random split is there to show the gap.
Position columns are left out: Elliptic has no edges between time steps, so its graph breaks into
many separate parts and global Laplacian positions mean little.
"""
import argparse
import json
import pathlib
import sys

import numpy as np
import pandas as pd
import scipy.sparse as sp
import xgboost as xgb
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from graph2table.evaluate import scores, summarize  # noqa: E402
from graph2table.features import flatten_graph  # noqa: E402

LAST_TRAIN_STEP = 34


def load(folder):
    folder = pathlib.Path(folder)
    features = pd.read_csv(folder / "elliptic_txs_features.csv", header=None)
    classes = pd.read_csv(folder / "elliptic_txs_classes.csv")
    edges = pd.read_csv(folder / "elliptic_txs_edgelist.csv")
    ids = features[0].to_numpy()
    index = pd.Series(np.arange(len(ids)), index=ids)
    steps = features[1].to_numpy()
    x = features.iloc[:, 2:].to_numpy(dtype=np.float64)
    labels = classes.set_index("txId").loc[ids, "class"].astype(str).to_numpy()
    y = np.where(labels == "1", 1, np.where(labels == "2", 0, -1))
    src = index.loc[edges["txId1"]].to_numpy()
    dst = index.loc[edges["txId2"]].to_numpy()
    adj = sp.coo_matrix((np.ones(len(src)), (src, dst)), shape=(len(ids), len(ids))).tocsr()
    return adj, x, y, steps


def fit_score(table, y, train, test, seed):
    model = xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.1, max_depth=6, subsample=0.8, colsample_bytree=0.8,
        tree_method="hist", random_state=seed, n_jobs=4,
    )
    model.fit(table[train], y[train])
    return scores(y[test], model.predict_proba(table[test])[:, 1])


VARIANTS = {
    "raw": ("raw",),
    "raw+structure": ("raw", "structure"),
    "raw+neighbours_mean": ("raw", "neighbours"),
    "raw+structure+neighbours_mean": ("raw", "structure", "neighbours"),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    adj, x, y, steps = load(args.data)
    labelled = np.flatnonzero(y >= 0)
    time_train = labelled[steps[labelled] <= LAST_TRAIN_STEP]
    time_test = labelled[steps[labelled] > LAST_TRAIN_STEP]
    results = {
        "labelled": int(len(labelled)),
        "time_split": {"train": int(len(time_train)), "test": int(len(time_test)),
                       "train_fraud_rate": float(y[time_train].mean()), "test_fraud_rate": float(y[time_test].mean())},
    }
    print(results, flush=True)
    for name, blocks in VARIANTS.items():
        table, _ = flatten_graph(adj, x, blocks=blocks)
        time_rows, random_rows = [], []
        for seed in range(args.seeds):
            time_rows.append(fit_score(table, y, time_train, time_test, seed))
            r_train, r_test = train_test_split(labelled, train_size=len(time_train), stratify=y[labelled], random_state=seed)
            random_rows.append(fit_score(table, y, r_train, r_test, seed))
        results[name] = {"columns": table.shape[1], "seeds": args.seeds, "time": summarize(time_rows), "random": summarize(random_rows)}
        print(name, results[name], flush=True)
        json.dump(results, open(args.out, "w"), indent=1)


if __name__ == "__main__":
    main()
