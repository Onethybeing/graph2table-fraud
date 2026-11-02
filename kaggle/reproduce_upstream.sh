#!/bin/bash
# Reproduce the LimiX numbers with the original TFM4GAD code, inside a Kaggle GPU kernel.
#
# The original repo has no licence, so its code is not copied into this one. This script fetches it
# at the pinned commits we tested. These are the LimiX steps from our tested kernel.
# The LimiX weights are for academic research; commercial use needs authorisation from their owners.
TFM4GAD_COMMIT=4bb8a5147a8589fe775ea8aa610d0e5a18ab0614
LIMIX_COMMIT=a50777da8d41607bc6ea467c7e16b4e051c1fb19
W=/kaggle/working; T=/tmp/w; mkdir -p $T $W/logs
set -e
pip install -q uv gdown
uv venv --seed --python 3.11 --python-preference only-managed $T/venv
P=$T/venv/bin/python
$P -m pip install -q torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
$P -m pip install -q dgl==2.4.0+cu124 -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
$P -m pip install -q nvidia-cusparse-cu12 nvidia-cublas-cu12 nvidia-cuda-runtime-cu12 nvidia-cuda-nvrtc-cu12
export LD_LIBRARY_PATH=$(find $T/venv -name 'lib*.so.12*' -printf '%h\n' | sort -u | tr '\n' ':')$LD_LIBRARY_PATH
$P -m pip install -q torchdata==0.9.0 numpy==1.26.4 pandas scipy==1.13.1 scikit-learn pyyaml pydantic sympy tqdm einops kditransform hyperopt xgboost huggingface_hub
cd $T
git clone -q https://github.com/Cloudy1225/TFM4GAD.git; git -C TFM4GAD checkout -q $TFM4GAD_COMMIT
git clone -q https://github.com/limix-ldm/LimiX.git limix_src; git -C limix_src checkout -q $LIMIX_COMMIT
cp -rn limix_src/. TFM4GAD/LimiX/; rm -rf TFM4GAD/LimiX/.git
mkdir -p TFM4GAD/LimiX/ckpts TFM4GAD/datasets
cd TFM4GAD/datasets
gdown 1txzXrzwBBAOEATXmfKzMUUKaXh6PJeR1 -O gad.zip
f=$(unzip -Z1 gad.zip | grep -E '(^|/)tfinance$' | head -n 1); unzip -q -o -j gad.zip "$f" -d .; rm -f gad.zip
cd ../LimiX/tfm4gad
export DGLBACKEND=pytorch
for model in LimiX-16M LimiX-2M; do
  $P -c "from huggingface_hub import hf_hub_download as d; import sys; d(f'stableai-org/{sys.argv[1]}', f'{sys.argv[1]}.ckpt', local_dir='../ckpts', token=False)" "$model"
  $P main.py --dataset tfinance --model_ckpt "../ckpts/$model.ckpt" > "$W/logs/$model.log" 2>&1
  grep -E "AUROC=" "$W/logs/$model.log" | tail -n 1
done
