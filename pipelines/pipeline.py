"""SageMaker Pipeline (the CI/CD DAG) for the CloudTrail risk model.

    ValidateAndPreprocess -> TrainModel -> EvaluateModel -> QualityGate
                                                              |-- pass -> RegisterModel (PendingManualApproval)
                                                              '-- fail -> QualityGateFailed (pipeline fails)

Run from notebook 06, or from a terminal inside SageMaker Studio:
    python pipelines/pipeline.py --role-arn <LabRole ARN> --start
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import sagemaker
from sagemaker.estimator import Estimator
from sagemaker.inputs import TrainingInput
from sagemaker.model import Model
from sagemaker.model_metrics import MetricsSource, ModelMetrics
from sagemaker.processing import FrameworkProcessor, ProcessingInput, ProcessingOutput
from sagemaker.sklearn.estimator import SKLearn
from sagemaker.workflow.condition_step import ConditionStep
from sagemaker.workflow.conditions import ConditionGreaterThanOrEqualTo
from sagemaker.workflow.fail_step import FailStep
from sagemaker.workflow.functions import Join, JsonGet
from sagemaker.workflow.model_step import ModelStep
from sagemaker.workflow.parameters import ParameterFloat, ParameterString
from sagemaker.workflow.pipeline import Pipeline
from sagemaker.workflow.pipeline_context import PipelineSession
from sagemaker.workflow.properties import PropertyFile
from sagemaker.workflow.steps import ProcessingStep, TrainingStep

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))
from cloudtrail_mlops import config  # noqa: E402

PIPELINE_NAME = "cloudtrail-risk-pipeline"
MODEL_PACKAGE_GROUP = "cloudtrail-risk-models"
SKLEARN_VERSION = "1.2-1"
BEST_PARAMS_FILE = REPO_ROOT / "pipelines" / "best_hyperparameters.json"


def stage_source_dir() -> Path:
    """Copy the step scripts + the shared package into one folder that is uploaded to the jobs."""
    staging = Path(tempfile.mkdtemp(prefix="ct_pipeline_src_"))
    for script in (REPO_ROOT / "pipelines" / "steps").glob("*.py"):
        shutil.copy(script, staging / script.name)
    shutil.copytree(REPO_ROOT / "src" / "cloudtrail_mlops", staging / "cloudtrail_mlops",
                    ignore=shutil.ignore_patterns("__pycache__"))
    # evaluate.py loads the model with the same XGBoost version the training container uses
    (staging / "requirements.txt").write_text("xgboost==1.7.6\n")
    return staging


def hyperparameters() -> dict:
    """Tuned hyper-parameters from notebook 03 if available, else the defaults in config."""
    params = dict(config.DEFAULT_HYPERPARAMETERS)
    if BEST_PARAMS_FILE.exists():
        params.update(json.loads(BEST_PARAMS_FILE.read_text()))
    return params


def get_pipeline(role: str, bucket: str, instance_type: str = "ml.m5.large") -> Pipeline:
    session = PipelineSession()
    region = session.boto_region_name
    s3_root = f"s3://{bucket}/{config.S3_PREFIX}/{config.S3_LAYOUT['pipeline']}"

    # ---- parameters (can be overridden per execution) ---------------------------------
    p_input = ParameterString("FeatureMartUri",
                              default_value=f"s3://{bucket}/{config.S3_PREFIX}/{config.S3_LAYOUT['features']}/csv/")
    p_min_auc = ParameterFloat("MinTestRocAuc", default_value=config.MIN_TEST_ROC_AUC)
    p_min_lift = ParameterFloat("MinTestApLift", default_value=config.MIN_TEST_AP_LIFT)
    p_approval = ParameterString("ModelApprovalStatus", default_value="PendingManualApproval")

    source_dir = str(stage_source_dir())

    def processor(base_job_name: str) -> FrameworkProcessor:
        return FrameworkProcessor(estimator_cls=SKLearn, framework_version=SKLEARN_VERSION, role=role,
                                  instance_type=instance_type, instance_count=1,
                                  base_job_name=base_job_name, sagemaker_session=session)

    # ---- 1. validate + preprocess -----------------------------------------------------
    step_preprocess = ProcessingStep(
        name="ValidateAndPreprocess",
        step_args=processor("ct-preprocess").run(
            code="preprocess.py", source_dir=source_dir,
            inputs=[ProcessingInput(source=p_input, destination="/opt/ml/processing/input")],
            outputs=[ProcessingOutput(output_name="model_input", source="/opt/ml/processing/output")],
        ),
    )
    model_input_uri = step_preprocess.properties.ProcessingOutputConfig.Outputs["model_input"].S3Output.S3Uri

    # ---- 2. train (SageMaker built-in XGBoost) ----------------------------------------
    image_uri = sagemaker.image_uris.retrieve("xgboost", region=region, version=config.XGBOOST_VERSION)
    estimator = Estimator(image_uri=image_uri, role=role, instance_type=instance_type, instance_count=1,
                          output_path=f"{s3_root}/models", base_job_name="ct-train",
                          hyperparameters=hyperparameters(), sagemaker_session=session)
    step_train = TrainingStep(
        name="TrainModel",
        step_args=estimator.fit({
            "train": TrainingInput(Join(on="/", values=[model_input_uri, "train"]), content_type="text/csv"),
            "validation": TrainingInput(Join(on="/", values=[model_input_uri, "validation"]), content_type="text/csv"),
        }),
    )

    # ---- 3. evaluate on validation (thresholds) and test (metrics) ---------------------
    evaluation_report = PropertyFile(name="EvaluationReport", output_name="evaluation", path="evaluation.json")
    step_evaluate = ProcessingStep(
        name="EvaluateModel",
        step_args=processor("ct-evaluate").run(
            code="evaluate.py", source_dir=source_dir,
            inputs=[ProcessingInput(source=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                                    destination="/opt/ml/processing/model"),
                    ProcessingInput(source=model_input_uri, destination="/opt/ml/processing/data")],
            outputs=[ProcessingOutput(output_name="evaluation", source="/opt/ml/processing/evaluation")],
        ),
        property_files=[evaluation_report],
    )
    evaluation_uri = Join(on="/", values=[
        step_evaluate.properties.ProcessingOutputConfig.Outputs["evaluation"].S3Output.S3Uri, "evaluation.json"])

    # ---- 4a. register the model (only reached when the gate passes) --------------------
    model = Model(image_uri=image_uri, model_data=step_train.properties.ModelArtifacts.S3ModelArtifacts,
                  role=role, sagemaker_session=session)
    step_register = ModelStep(
        name="RegisterModel",
        step_args=model.register(
            content_types=["text/csv"], response_types=["text/csv"],
            inference_instances=["ml.t2.medium", "ml.m5.large"], transform_instances=["ml.m5.large"],
            model_package_group_name=MODEL_PACKAGE_GROUP, approval_status=p_approval,
            model_metrics=ModelMetrics(model_statistics=MetricsSource(s3_uri=evaluation_uri,
                                                                      content_type="application/json")),
            description="CloudTrail actor x 5-minute window risk model (XGBoost), registered by the pipeline",
        ),
    )

    # ---- 4b. stop the pipeline with a clear message when the gate fails ----------------
    step_fail = FailStep(
        name="QualityGateFailed",
        error_message=Join(on=" ", values=["Model rejected: test ROC-AUC must be >=", p_min_auc,
                                           "and test AP lift must be >=", p_min_lift]),
    )

    # ---- 4. quality gate --------------------------------------------------------------
    step_gate = ConditionStep(
        name="QualityGate",
        conditions=[
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(step_name=step_evaluate.name, property_file=evaluation_report, json_path="metrics.test_roc_auc"),
                right=p_min_auc),
            ConditionGreaterThanOrEqualTo(
                left=JsonGet(step_name=step_evaluate.name, property_file=evaluation_report, json_path="metrics.test_ap_lift"),
                right=p_min_lift),
        ],
        if_steps=[step_register],
        else_steps=[step_fail],
    )

    return Pipeline(name=PIPELINE_NAME, parameters=[p_input, p_min_auc, p_min_lift, p_approval],
                    steps=[step_preprocess, step_train, step_evaluate, step_gate], sagemaker_session=session)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create/update the pipeline and optionally start it")
    parser.add_argument("--role-arn", required=True)
    parser.add_argument("--bucket", default=None, help="defaults to the SageMaker default bucket")
    parser.add_argument("--start", action="store_true")
    args = parser.parse_args()
    bucket = args.bucket or sagemaker.Session().default_bucket()
    pipeline = get_pipeline(role=args.role_arn, bucket=bucket)
    pipeline.upsert(role_arn=args.role_arn)
    print("Pipeline upserted:", PIPELINE_NAME)
    if args.start:
        print("Started execution:", pipeline.start().arn)
