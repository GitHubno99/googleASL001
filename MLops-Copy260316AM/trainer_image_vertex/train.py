# Copyright 2021 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#            http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Regressor trainer script for the specified dataset using GaussianProcessRegressor."""

import os
import pickle
import subprocess
import sys

import fire
import hypertune
import pandas as pd
from sklearn.compose import ColumnTransformer
# GaussianProcessRegressor と必要なカーネルをインポート
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# AIP_MODEL_DIRはGoogle Cloud AI Platformで実行する際に環境変数として設定されます
AIP_MODEL_DIR = os.environ.get("AIP_MODEL_DIR", "./")
MODEL_FILENAME = "model.pkl"


def train_evaluate(
    training_dataset_path, validation_dataset_path, alpha, hptune
):
    """Trains the Gaussian Process Regressor model."""
    # GPRでは学習率ではなく、観測ノイズの強さを表す alpha が重要なパラメータになります
    alpha = float(alpha)

    df_train = pd.read_csv(training_dataset_path)
    df_validation = pd.read_csv(validation_dataset_path)

    if not hptune:
        df_train = pd.concat([df_train, df_validation])

    # 説明変数をX1からX8までの8列に指定 [cite: 99]
    numeric_feature_indexes = slice(0, 8)

    # 数値特徴量の標準化
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_feature_indexes),
        ]
    )

    # ガウス過程回帰のカーネル定義
    # ConstantKernel (C) * RBFカーネル を基本構成とします
    kernel = C(1.0, (1e-3, 1e3)) * RBF(1.0, (1e-2, 1e2))

    # パイプラインを定義 (前処理 + GPR回帰モデル)
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            n_restarts_optimizer=20, 
            normalize_y=True,        
            random_state=42
        ))
    ])

    # 数値特徴量のデータ型をfloat64に設定 [cite: 102]
    num_features_type_map = {
        feature: "float64"
        for feature in df_train.columns[numeric_feature_indexes]
    }
    df_train = df_train.astype(num_features_type_map)
    df_validation = df_validation.astype(num_features_type_map)

    print(f"Starting training GPR: alpha={alpha}")

    # 目的変数を 'Y1' に設定し、説明変数から 'Y1' と 'Y2' を除外
    X_train = df_train.drop(["Y1", "Y2"], axis=1)
    y_train = df_train["Y1"]

    # モデルの学習を実行
    pipeline.fit(X_train, y_train)

    # ハイパーパラメータチューニングの場合、モデルの評価を行う [cite: 103]
    if hptune:
        X_validation = df_validation.drop(["Y1", "Y2"], axis=1)
        y_validation = df_validation["Y1"]
        
        # 評価指標としてR2スコア（決定係数）を使用 [cite: 103]
        score = pipeline.score(X_validation, y_validation)
        print(f"Model R^2 score: {score}")

        # hypertuneサービスに評価指標を報告 [cite: 104]
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag="r2_score", metric_value=score
        )

    # ハイパーパラメータチューニングでない場合（最終学習）、モデルを保存
    if not hptune:
        with open(MODEL_FILENAME, "wb") as model_file:
            pickle.dump(pipeline, model_file)
        
        if "gs://" in AIP_MODEL_DIR:
            # ★ 修正ポイント：保存先を「ファイル名」まで明示的に結合する
            model_path = os.path.join(AIP_MODEL_DIR, MODEL_FILENAME)
            subprocess.check_call(
                ["gsutil", "cp", MODEL_FILENAME, model_path], stderr=sys.stdout
            )
            print(f"Saved model in: {model_path}")
        else:
            print(f"Saved model locally at: {MODEL_FILENAME}")


if __name__ == "__main__":
    fire.Fire(train_evaluate)
