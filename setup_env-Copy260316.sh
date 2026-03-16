#!/bin/bash
# -*- coding: utf-8 -*-

# 1. 環境名の設定
ENV_NAME="py311_adk"
PYTHON_VERSION="3.11"

echo "--- Starting Environment Setup: $ENV_NAME (Python $PYTHON_VERSION) ---"

# 2. 既存の環境があれば削除（クリーンインストール用）
conda remove -n $ENV_NAME --all -y

# 3. Python 3.11 環境の作成
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# 4. コンダ環境をアクティベートしてライブラリをインストール
# ※シェルスクリプト内では 'source' を使用します
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

echo "--- Installing Libraries ---"
pip install ipykernel google-adk google-cloud-aiplatform cloudpickle pydantic scikit-learn==1.2.2 pandas

# 5. JupyterLabのカーネルとして登録
echo "--- Registering Jupyter Kernel ---"
python -m ipykernel install --user --name $ENV_NAME --display-name "Python 3.11 (ADK_Auto)"

echo "--- Setup Complete! Please refresh your JupyterLab and select '$ENV_NAME' kernel. ---"
