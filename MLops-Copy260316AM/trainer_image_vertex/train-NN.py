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

"""Regressor trainer script for the specified dataset using MLPRegressor."""

import os
import pickle
import subprocess
import sys

import fire
import hypertune
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# AIP_MODEL_DIRはGoogle Cloud AI Platformで実行する際に環境変数として設定されます
AIP_MODEL_DIR = os.environ.get("AIP_MODEL_DIR", "./")
MODEL_FILENAME = "model.pkl"


def train_evaluate(
    training_dataset_path, validation_dataset_path, alpha, max_iter, hptune
):
    """Trains the MLP Regressor model."""
    max_iter = int(float(max_iter))
    alpha = float(alpha)

    df_train = pd.read_csv(training_dataset_path)
    df_validation = pd.read_csv(validation_dataset_path)

    if not hptune:
        df_train = pd.concat([df_train, df_validation])

    # 説明変数をX1からX8までの8列に指定
    numeric_feature_indexes = slice(0, 8)

    # 数値特徴量のみを標準化する前処理を定義
    # ニューラルネットワーク（MLP）において標準化は非常に重要です
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_feature_indexes),
        ]
    )

    # パイプラインを定義 (前処理 + MLP回帰モデル)
    pipeline = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("regressor", MLPRegressor(
                hidden_layer_sizes=(32, 64),#シンプルな構成
                activation='relu',         # 活性化関数
                solver='adam',             # 小〜中規模データに適した最適化手法
                random_state=42,           # 結果の再現性のため
                # early_stopping=False        # 過学習防止のため検証データで早期終了
            ))
        ]
    )

    # 数値特徴量のデータ型をfloat64に設定
    num_features_type_map = {
        feature: "float64"
        for feature in df_train.columns[numeric_feature_indexes]
    }
    df_train = df_train.astype(num_features_type_map)
    df_validation = df_validation.astype(num_features_type_map)

    print(f"Starting training MLP: alpha={alpha}, max_iter={max_iter}")

    # 目的変数を 'Y1' に設定し、説明変数から 'Y1' と 'Y2' を除外
    X_train = df_train.drop(["Y1", "Y2"], axis=1)
    y_train = df_train["Y1"]

    # ハイパーパラメータを設定
    pipeline.set_params(
        regressor__learning_rate_init=alpha,  # 引数のalphaを学習率として割り当てる
        regressor__max_iter=max_iter
    )
    
    # モデルの学習を実行
    pipeline.fit(X_train, y_train)

    # ハイパーパラメータチューニングの場合、モデルの評価を行う
    if hptune:
        X_validation = df_validation.drop(["Y1", "Y2"], axis=1)
        y_validation = df_validation["Y1"]
        
        # 評価指標としてR2スコア（決定係数）を使用
        score = pipeline.score(X_validation, y_validation)
        print(f"Model R^2 score: {score}")

        # hypertuneサービスに評価指標を報告
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag="r2_score", metric_value=score
        )

    # ハイパーパラメータチューニングでない場合（最終学習）、モデルを保存
    if not hptune:
        # ローカルに一時的にモデルファイルを保存
        with open(MODEL_FILENAME, "wb") as model_file:
            pickle.dump(pipeline, model_file)
        
        # gsutilコマンドでGCSにアップロード
        if "gs://" in AIP_MODEL_DIR:
            subprocess.check_call(
                ["gsutil", "cp", MODEL_FILENAME, AIP_MODEL_DIR], stderr=sys.stdout
            )
            print(f"Saved model in: {AIP_MODEL_DIR}")
        else:
            print(f"Saved model locally at: {MODEL_FILENAME}")


if __name__ == "__main__":
    fire.Fire(train_evaluate)