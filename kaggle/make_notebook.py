"""Build a private Kaggle kernel that runs this repo's code on T-Finance and Elliptic.

Usage: python kaggle/make_notebook.py <kaggle-username>, then: kaggle kernels push -p kaggle/build
No token is needed: all data here is public. CPU only.
"""
import json
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
BUILD = ROOT / "kaggle" / "build"
SLUG = "graph2table-fraud-run"
FILES = ["graph2table/__init__.py", "graph2table/features.py", "graph2table/evaluate.py",
         "scripts/run_tfinance.py", "scripts/run_elliptic.py", "tests_features.py"]

RUN = r"""#!/bin/bash
W=/kaggle/working; T=/tmp/w; mkdir -p $T
cd $W/repo
pip install -q uv gdown
uv venv --seed --python 3.11 --python-preference only-managed $T/venv
P=$T/venv/bin/python
$P -m pip install -q torch==2.4.0 --index-url https://download.pytorch.org/whl/cpu
$P -m pip install -q dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/repo.html
$P -m pip install -q torchdata==0.9.0 numpy==1.26.4 scipy==1.13.1 pandas scikit-learn xgboost pydantic pyyaml
export DGLBACKEND=pytorch
$P tests_features.py
for i in 1 2 3; do gdown 1txzXrzwBBAOEATXmfKzMUUKaXh6PJeR1 -O $T/gad.zip 2>&1 | tail -n 2; [ -s $T/gad.zip ] && break; sleep 60; done
f=$(unzip -Z1 $T/gad.zip | grep -E '(^|/)tfinance$' | head -n 1); unzip -q -o -j $T/gad.zip "$f" -d $T; rm -f $T/gad.zip
$P scripts/run_tfinance.py --graph $T/tfinance --out $W/tfinance.json 2>&1 | tee $W/tfinance.log
D=$(dirname "$(find /kaggle/input -name elliptic_txs_features.csv | head -n 1)")
$P scripts/run_elliptic.py --data "$D" --out $W/elliptic.json 2>&1 | tee $W/elliptic.log
rm -rf $W/repo
echo DONE
"""


def code(text):
    return {"cell_type": "code", "metadata": {}, "outputs": [], "execution_count": None, "source": text.splitlines(True)}


def main():
    user = sys.argv[1]
    BUILD.mkdir(parents=True, exist_ok=True)
    contents = {name: (ROOT / name).read_text(encoding="utf-8") for name in FILES}
    writer = (
        "import pathlib\n"
        f"files = {contents!r}\n"
        "for name, text in files.items():\n"
        "    path = pathlib.Path('/kaggle/working/repo') / name\n"
        "    path.parent.mkdir(parents=True, exist_ok=True)\n"
        "    path.write_text(text)\n"
        f"pathlib.Path('/kaggle/working/run.sh').write_text({RUN!r})\n"
    )
    notebook = {
        "cells": [code(writer), code("!bash /kaggle/working/run.sh 2>&1 | tail -n 80\n")],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    (BUILD / f"{SLUG}.ipynb").write_text(json.dumps(notebook), encoding="utf-8")
    metadata = {
        "id": f"{user}/{SLUG}", "title": SLUG, "code_file": f"{SLUG}.ipynb", "language": "python",
        "kernel_type": "notebook", "is_private": True, "enable_gpu": False, "enable_internet": True,
        "dataset_sources": ["ellipticco/elliptic-data-set"], "competition_sources": [], "kernel_sources": [],
    }
    (BUILD / "kernel-metadata.json").write_text(json.dumps(metadata, indent=1), encoding="utf-8")
    print("wrote", BUILD)


if __name__ == "__main__":
    main()
