# 🔗Fusing Kubeflow Pipelines and TFX Components on Vertex AI

**By Pallav Anand**

---

## Overview

Kubeflow Pipelines (KFP) gives you flexibility. TensorFlow Extended (TFX) gives you battle-tested components for data validation, model evaluation, and governance. Why choose?

Vertex AI Pipelines supports both natively. This tutorial shows how to combine them in a single pipeline:

- 🧩**Custom KFP Components** — Write load, transform, and train steps your way
- ✅**TFX Data Validation** — Use `StatisticsGen` and `ExampleValidator` to detect data drift
- 📊**TFX Model Evaluation** — Apply threshold-based model blessing with `Evaluator`
- 🔗**Fused Pipeline** — Wire it all together in a single `@dsl.pipeline`
- 🚀**Run on Vertex AI** — Submit the hybrid pipeline with one command

Find the code on my GitHub.

---

## 🎯Why Fuse KFP and TFX?

| Capability | KFP Custom | TFX-Powered |
|---|---|---|
| Custom business logic | ✅ Easy | ❌ Complex |
| Data validation out of the box | ❌ Build from scratch | ✅ StatisticsGen + ExampleValidator |
| Model evaluation with thresholds | ❌ Build from scratch | ✅ Evaluator with blessed model |
| Type-safe artifacts and lineage | ✅ Built-in | ✅ ML Metadata |
| Container control | ✅ Full control | ✅ Configurable |

TFX's data validation and model evaluation are production-hardened — anomaly detection, skew detection, threshold-based model blessing. Building these from scratch in raw KFP takes weeks. Using TFX-powered components via `google-cloud-pipeline-components`, you get them as drop-in KFP components.

---

## 🛠Setup

### Install dependencies

```bash
pip install kfp google-cloud-pipeline-components google-cloud-aiplatform \
            pandas scikit-learn joblib
```

### GCP prerequisites

- Vertex AI API enabled
- A GCS bucket for pipeline artifacts
- Service account with Vertex AI User + Storage Object Admin roles

---

## 🧩Step 1: Custom KFP Components

We define two custom components: one to load data from BigQuery, one to train a model.

### Load Data Component

```python
from kfp.dsl import component, Output, Dataset, Input

@component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-bigquery==3.24.0", "pandas==2.0.3"],
)
def load_data_op(
    query: str,
    project_id: str,
    dataset: Output[Dataset],
) -> int:
    import pandas as pd
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)
    df = client.query(query).to_dataframe()
    df.to_csv(dataset.path, index=False)
    return len(df)
```

### Train Model Component

```python
@component(
    base_image="python:3.10",
    packages_to_install=["pandas==2.0.3", "scikit-learn==1.3.0", "joblib==1.3.2"],
)
def train_model_op(
    dataset: Input[Dataset],
    model: Output[Model],
    target_col: str = "target",
    test_size: float = 0.2,
) -> float:
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    import joblib

    df = pd.read_csv(dataset.path)
    X = df.drop(target_col, axis=1)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)

    score = accuracy_score(y_test, clf.predict(X_test))
    joblib.dump(clf, model.path)

    model.metadata["accuracy"] = score
    return score
```

These are standard KFP components. Nothing TFX-specific yet. They run in their own containers and produce typed artifacts.

---

## ✅Step 2: TFX-Powered Data Validation

TFX's `StatisticsGen` and `ExampleValidator` are wrapped as KFP components in `google-cloud-pipeline-components`. They generate data statistics and detect anomalies without writing a single validation rule.

### Statistics Generation

```python
from google_cloud_pipeline_components.v1.data_validation import (
    DataValidationOp,
    DataValidationMetricsOp,
)
```

These components compute descriptive statistics (min, max, mean, quantiles, null counts) for every feature. The output feeds into the anomaly detector.

### Example Validation

TFX compares incoming data statistics against a previous baseline and flags:

- **Feature anomalies**: New categories in categorical features
- **Value anomalies**: Out-of-range numeric values
- **Distribution skew**: Significant distribution changes

For this tutorial, we'll use a lightweight wrapper that runs TFX's validation logic:

```python
@component(
    base_image="python:3.10",
    packages_to_install=["tensorflow-data-validation==1.15.0"],
)
def validate_data_op(
    dataset: Input[Dataset],
    validation_result: Output[Artifact],
) -> NamedTuple("Outputs", [("anomalies_detected", int), ("is_valid", bool)]):
    import tensorflow_data_validation as tfdv
    import pandas as pd

    df = pd.read_csv(dataset.path)
    stats = tfdv.generate_statistics_from_dataframe(df)
    schema = tfdv.infer_schema(stats)

    # Detect anomalies
    anomalies = tfdv.validate_statistics(stats, schema)

    num_anomalies = len(anomalies.anomaly_info)
    is_valid = num_anomalies == 0

    return num_anomalies, is_valid
```

**Note:** In production, you'd save the schema and compare against it on subsequent runs. For simplicity, we infer it on the fly.

---

## 📊Step 3: TFX-Powered Model Evaluation

TFX's `Evaluator` blesses or rejects models based on configured thresholds. We wrap it as a KFP component:

```python
@component(
    base_image="python:3.10",
    packages_to_install=[
        "tensorflow==2.15.0",
        "tensorflow-model-analysis==0.46.0",
    ],
)
def evaluate_model_op(
    model: Input[Model],
    dataset: Input[Dataset],
    eval_result: Output[Artifact],
    accuracy_threshold: float = 0.7,
) -> NamedTuple("Outputs", [("is_blessed", bool), ("accuracy", float)]):
    import pandas as pd
    import joblib
    from sklearn.metrics import accuracy_score

    df = pd.read_csv(dataset.path)
    X = df.drop("target", axis=1)
    y = df["target"]

    clf = joblib.load(model.path)
    y_pred = clf.predict(X)
    acc = accuracy_score(y, y_pred)

    is_blessed = acc >= accuracy_threshold

    return is_blessed, acc
```

The `is_blessed` flag acts as a gate: downstream components check it before deploying. This is the **Champion/Challenger** pattern — only models above the threshold reach production.

---

## 🔗Step 4: The Fused Pipeline

Now we combine custom KFP components and TFX-powered components in a single `@dsl.pipeline`:

```python
from kfp import dsl

@dsl.pipeline(
    name="hybrid-ml-pipeline",
    description="KFP + TFX fused pipeline",
)
def hybrid_pipeline(
    query: str = "SELECT * FROM `project.dataset.table`",
    project_id: str = "",
    accuracy_threshold: float = 0.7,
):
    # Step 1: Load data (custom KFP component)
    load_task = load_data_op(query=query, project_id=project_id)

    # Step 2: Validate data (TFX-powered component)
    validate_task = validate_data_op(dataset=load_task.outputs["dataset"])

    # Step 3: Gate on data validity
    with dsl.Condition(validate_task.outputs["is_valid"] == True, name="data-valid"):

        # Step 4: Train model (custom KFP component)
        train_task = train_model_op(
            dataset=load_task.outputs["dataset"],
        )

        # Step 5: Evaluate model (TFX-powered component)
        eval_task = evaluate_model_op(
            model=train_task.outputs["model"],
            dataset=load_task.outputs["dataset"],
            accuracy_threshold=accuracy_threshold,
        )

        # Step 6: Deploy if blessed
        with dsl.Condition(eval_task.outputs["is_blessed"] == True, name="model-blessed"):
            deploy_task = deploy_model_op(
                model=train_task.outputs["model"],
                project_id=project_id,
            )
```

**Key fusion points:**

| Step | Component Type | What It Does |
|---|---|---|
| `load_task` | Custom KFP | Query BigQuery, produce Dataset artifact |
| `validate_task` | TFX-powered | Statistics + anomaly detection |
| `train_task` | Custom KFP | Train scikit-learn model |
| `eval_task` | TFX-powered | Threshold-based model blessing |
| `deploy_task` | Custom KFP | Deploy to Vertex AI Endpoint |

The pipeline uses KFP's native `Condition` for branching — no framework glue needed.

### Deploy Component

```python
@component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-aiplatform==1.55.0"],
)
def deploy_model_op(
    model: Input[Model],
    project_id: str,
    endpoint_name: str = "hybrid-model-endpoint",
) -> str:
    from google.cloud import aiplatform

    aiplatform.init(project=project_id)

    uploaded_model = aiplatform.Model.upload(
        display_name="hybrid-model",
        artifact_uri=model.uri,
        serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
    )

    endpoint = aiplatform.Endpoint.create(
        display_name=endpoint_name,
    )

    deployment = endpoint.deploy(
        model=uploaded_model,
        machine_type="n1-standard-2",
    )

    return endpoint.resource_name
```

---

## 🚀Step 5: Run on Vertex AI

### Compile the pipeline

```bash
kfp compiler compile --pipeline-file pipeline.py --output pipeline.yaml
```

### Submit to Vertex AI

```bash
# Using gcloud
gcloud ai pipelines run \
  --project=PROJECT_ID \
  --region=us-central1 \
  --display-name=hybrid-pipeline-run \
  --pipeline-file=pipeline.yaml \
  --parameter="project_id=PROJECT_ID" \
  --parameter="query=SELECT * FROM PROJECT.dataset.table"

# Or using Python SDK
from google.cloud import aiplatform

aiplatform.init(project="PROJECT_ID", location="us-central1")

job = aiplatform.PipelineJob(
    display_name="hybrid-pipeline-run",
    template_path="pipeline.yaml",
    parameter_values={
        "query": "SELECT * FROM `project.dataset.table`",
        "project_id": "PROJECT_ID",
        "accuracy_threshold": 0.7,
    },
)

job.run()
```

### Monitor in Vertex AI UI

1. Go to **Vertex AI → Pipelines**
2. Click on your run to see the DAG
3. Each component shows:
   - Execution status and duration
   - Input/output artifacts
   - Logs and errors
   - TFX components show additional metadata (statistics, anomalies, blessing)

---

## 📋Step 6: Inspect Results

### Check data validation output

```python
import tensorflow_data_validation as tfdv

# Load validation result
anomalies = tfdv.load_anomalies_text("path/to/validation_result")
for feature, info in anomalies.anomaly_info.items():
    print(f"{feature}: {info.short_description}")
```

### Check model blessing status

```python
# From pipeline output
print(f"Model blessed: {eval_task.outputs['is_blessed']}")
print(f"Accuracy: {eval_task.outputs['accuracy']}")
```

### Verify endpoint

```bash
gcloud ai endpoints list --region=us-central1
```

---

## ❗Common Issues

### Issue: "TFX and KFP version conflicts"

**Error:** `ImportError: cannot import name 'DataValidationOp'`

**Solution:** Pin compatible versions:

```bash
pip install kfp==2.7.0 google-cloud-pipeline-components==2.8.0
```

Check the [compatibility matrix](https://cloud.google.com/vertex-ai/docs/pipelines/version-compatibility) for the latest.

### Issue: "Dataset artifact not found by TFX component"

**Error:** `FileNotFoundError: ...`

**Solution:** TFX components expect data in TFRecord format. If using CSV, convert to TFRecord or use a custom TFX executor that reads CSV:

```python
@component(...)
def csv_to_tfrecord_op(dataset: Input[Dataset], output: Output[Dataset]):
    import tensorflow as tf
    import pandas as pd

    df = pd.read_csv(dataset.path)
    examples = [
        tf.train.Example(
            features=tf.train.Features(
                feature={col: tf.train.Feature(
                    float_list=tf.train.FloatList(value=[row[col]])
                ) for col in df.columns}
            )
        ).SerializeToString()
        for _, row in df.iterrows()
    ]

    with tf.io.TFRecordWriter(output.path) as writer:
        for ex in examples:
            writer.write(ex)
```

### Issue: "Pipeline compiles but no TFX components appear"

**Solution:** Ensure you're importing from the latest path:

```python
# Correct (v2+)
from google_cloud_pipeline_components.v1.data_validation import DataValidationOp

# Deprecated (v1)
from google_cloud_pipeline_components.experimental.tfx import DataValidationOp
```

### Issue: "Model deployment fails — no endpoint exists"

**Solution:** Check that the deploy component creates a new endpoint or accepts an existing endpoint resource name:

```python
endpoints = aiplatform.Endpoint.list(filter=f'display_name="{endpoint_name}"')
if endpoints:
    endpoint = endpoints[0]
else:
    endpoint = aiplatform.Endpoint.create(display_name=endpoint_name)
```

---

## 💡Key Takeaways

- **You don't have to choose** between KFP and TFX — Vertex AI Pipelines runs both
- **Custom KFP components** handle business logic (data loading, custom training)
- **TFX-powered components** provide production-hardened data validation and model evaluation
- **`dsl.Condition`** lets you gate pipeline flow on validation/evaluation results
- **The pipeline is still a single KFP DAG** — TFX components are drop-in replacements for custom ones

**When to use this pattern:**

| Scenario | Recommendation |
|---|---|
| Need flexible custom logic + data validation | ✅ Fused pipeline |
| Pure TensorFlow/Keras model training | ✅ Pure TFX pipeline |
| Non-TF models (XGBoost, sklearn, PyTorch) | ✅ Fused pipeline (custom train + TFX eval) |
| Simple inference-only pipeline | ✅ Pure KFP pipeline |

Let's connect on [LinkedIn](https://www.linkedin.com/in/pallav-anand-025526109/)!

**Tags:**
- Kubeflow Pipelines
- TensorFlow Extended
- Vertex AI
- MLOps
- Data Validation
- Google Cloud Platform
