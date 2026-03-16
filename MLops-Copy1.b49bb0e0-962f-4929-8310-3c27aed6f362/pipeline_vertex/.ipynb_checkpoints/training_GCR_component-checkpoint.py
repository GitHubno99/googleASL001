from kfp.dsl import component

@component(
    base_image="python:3.10", # ★ 安定したPython3.10に変更
    packages_to_install=["google-cloud-aiplatform"], # ★ 不要なライブラリを削除
)
def train_and_deploy(
    project: str,
    location: str,
    container_uri: str,
    serving_container_uri: str,
    training_file_path: str,
    validation_file_path: str,
    staging_bucket: str,
    alpha: float,
):
    from google.cloud import aiplatform
    import time

    aiplatform.init(project=project, location=location, staging_bucket=staging_bucket)

    job = aiplatform.CustomContainerTrainingJob(
        display_name="gpr_final_training",
        container_uri=container_uri,
        command=[
            "python", "train.py",
            f"--training_dataset_path={training_file_path}",
            f"--validation_dataset_path={validation_file_path}",
            f"--alpha={alpha}",
            "--hptune=False",
        ],
        model_serving_container_image_uri=serving_container_uri,
    )
    
    # ★ 修正ポイント：モデルの保存先フォルダを毎回被らないように生成する
    job_dir = f"{staging_bucket}/model_output_{int(time.time())}"
    model = job.run(
        replica_count=1, 
        model_display_name="gpr_kfp_model_v2",
        base_output_dir=job_dir  # 明示的に保存場所を指定
    )

    endpoint_name = "covertype_GCP_model_endpoint"
    endpoints = aiplatform.Endpoint.list(
        filter=f'display_name="{endpoint_name}"',
        order_by="create_time desc"
    )

    if endpoints:
        endpoint = endpoints[0]
    else:
        endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)

    model.deploy(
        endpoint=endpoint,
        machine_type="n1-standard-2",
        min_replica_count=1,
        max_replica_count=1,
        traffic_split={"0": 100},
        sync=True
    )