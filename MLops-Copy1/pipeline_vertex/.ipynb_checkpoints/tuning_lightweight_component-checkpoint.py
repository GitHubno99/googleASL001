"""Lightweight component tuning function."""
from typing import NamedTuple
from kfp.dsl import component

# ０３０８　　Scikit-learn 1.2対応
# @component(
#     base_image="us-docker.pkg.dev/vertex-ai/training/sklearn-cpu.1-0:latest",
#     packages_to_install=["google-cloud-aiplatform"],    
# )
@component(
    base_image="python:3.11",
    packages_to_install=["google-cloud-aiplatform", "scikit-learn==1.2.2", "pandas"],
)
def tune_hyperparameters(
    project: str,
    location: str,
    container_uri: str,
    training_file_path: str,
    validation_file_path: str,
    staging_bucket: str,
    max_trial_count: int,
    parallel_trial_count: int,
) -> NamedTuple(
    "Outputs",
    [("best_r2_score", float), ("best_alpha", float), ("best_max_iter", int)],
):
    # pylint: disable=import-outside-toplevel
    from google.cloud import aiplatform
    from google.cloud.aiplatform import hyperparameter_tuning as hpt

    aiplatform.init(
        project=project, location=location, staging_bucket=staging_bucket
    )

    worker_pool_specs = [
        {
            "machine_spec": {
                "machine_type": "n1-standard-4",
            },
            "replica_count": 1,
            "container_spec": {
                "image_uri": container_uri,
                "args": [
                    f"--training_dataset_path={training_file_path}",
                    f"--validation_dataset_path={validation_file_path}",
                    "--hptune",
                ],
            },
        }
    ]

    custom_job = aiplatform.CustomJob(
        display_name="covertype_kfp_trial_job",
        worker_pool_specs=worker_pool_specs,
    )

    hp_job = aiplatform.HyperparameterTuningJob(
        display_name="covertype_kfp_tuning_job",
        custom_job=custom_job,
        metric_spec={
            "r2_score": "maximize",
        },
        parameter_spec={
            "alpha": hpt.DoubleParameterSpec(
                min=1.0e-5, max=1.0e-2, scale="log"
            ),
            "max_iter": hpt.IntegerParameterSpec(
                min=100, max=1000, scale="linear"
            ),
        },
        max_trial_count=max_trial_count,
        parallel_trial_count=parallel_trial_count,
    )

    hp_job.run()
    
    best_r2_score = -float('inf')
    best_trial = None

    for trial in hp_job.trials:
        # 完了していない試行やメトリクスがない試行をスキップ
        if not hasattr(trial, 'final_measurement') or not trial.final_measurement:
            continue
            
        for metric in trial.final_measurement.metrics:
            if metric.metric_id == "r2_score":
                if metric.value > best_r2_score:
                    best_r2_score = metric.value
                    best_trial = trial

    if best_trial is None:
        raise RuntimeError("有効な r2_score を持つ試行が見つかりませんでした。学習ログを確認してください。")

    # パラメータも ID (名前) で検索して取得
    best_alpha = float(next(p.value for p in best_trial.parameters if p.parameter_id == 'alpha'))
    best_max_iter = int(next(p.value for p in best_trial.parameters if p.parameter_id == 'max_iter'))

    return (float(best_r2_score), float(best_alpha), int(best_max_iter))