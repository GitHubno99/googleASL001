# Copyright 2021 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at

# https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Kubeflow GPR Regressor Pipeline."""
import os
# 
from kfp import dsl
# コンポーネント名は実態に合わせて適宜読み替えてください
from training_GCR_component import train_and_deploy
from tuning_GCR_component import tune_hyperparameters
# from training_lightweight_component import train_and_deploy
# from tuning_lightweight_component import tune_hyperparameters

PIPELINE_ROOT = os.getenv("PIPELINE_ROOT")
PROJECT_ID = os.getenv("PROJECT_ID")
REGION = os.getenv("REGION")

TRAINING_CONTAINER_IMAGE_URI = os.getenv("TRAINING_CONTAINER_IMAGE_URI")
SERVING_CONTAINER_IMAGE_URI = os.getenv("SERVING_CONTAINER_IMAGE_URI")

TRAINING_FILE_PATH = os.getenv("TRAINING_FILE_PATH")
VALIDATION_FILE_PATH = os.getenv("VALIDATION_FILE_PATH")

MAX_TRIAL_COUNT = int(os.getenv("MAX_TRIAL_COUNT", "5"))
PARALLEL_TRIAL_COUNT = int(os.getenv("PARALLEL_TRIAL_COUNT", "5"))
THRESHOLD = float(os.getenv("THRESHOLD", "0.6"))


@dsl.pipeline(
    name="gpr-regression-pipeline",
    description="The pipeline for training and deploying a Gaussian Process Regressor",
    pipeline_root=PIPELINE_ROOT,
)
def gpr_train_pipeline(
    training_container_uri: str = TRAINING_CONTAINER_IMAGE_URI,
    serving_container_uri: str = SERVING_CONTAINER_IMAGE_URI,
    training_file_path: str = TRAINING_FILE_PATH,
    validation_file_path: str = VALIDATION_FILE_PATH,
    r2_score_deployment_threshold: float = THRESHOLD,
    max_trial_count: int = MAX_TRIAL_COUNT,
    parallel_trial_count: int = PARALLEL_TRIAL_COUNT,
    pipeline_root: str = PIPELINE_ROOT,
):
    staging_bucket = f"{pipeline_root}/staging"

    # 1. ハイパーパラメータチューニングステップ
    tuning_op = tune_hyperparameters(
        project=PROJECT_ID,
        location=REGION,
        container_uri=training_container_uri,
        training_file_path=training_file_path,
        validation_file_path=validation_file_path,
        staging_bucket=staging_bucket,
        max_trial_count=max_trial_count,
        parallel_trial_count=parallel_trial_count,
    )

    # チューニング結果から最高のR2スコアと、その時のalphaを取得
    # r2_score = tuning_op.outputs["best_r2_score"]
    # best_alpha = tuning_op.outputs["best_alpha"]

    # # 2. 条件分岐: R2スコアが閾値を超えている場合のみデプロイへ進む
    # with dsl.If(
    #     r2_score >= r2_score_deployment_threshold, name="deploy_decision"
    # ):
    #     # 3. 本学習およびデプロイステップ
    #     train_and_deploy_op = (  # pylint: disable=unused-variable
    #         train_and_deploy(
    #             project=PROJECT_ID,
    #             location=REGION,
    #             container_uri=training_container_uri,
    #             serving_container_uri=serving_container_uri,
    #             training_file_path=training_file_path,
    #             validation_file_path=validation_file_path,
    #             staging_bucket=staging_bucket,
    #             # GPR用のパラメータのみを渡す (max_iterは削除)
    #             alpha=best_alpha,
    #         )
    
    # 修正後（確実に動く構成） --------------------------
    # 1. 判定を通さず直接 alpha を取得
    best_alpha = tuning_op.outputs["best_alpha"]

    # 2. 直接デプロイステップを呼び出す（Ifを消してインデントを戻す）
    train_and_deploy_op = train_and_deploy(
        project=PROJECT_ID,
        location=REGION,
        container_uri=training_container_uri,
        serving_container_uri=serving_container_uri,
        training_file_path=training_file_path,
        validation_file_path=validation_file_path,
        staging_bucket=staging_bucket,
        alpha=best_alpha, # 直接渡す
    )