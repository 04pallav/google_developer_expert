# 🧪Testing Kubeflow Pipelines Locally: From Unit Tests to CI/CD

**By Pallav Anand**

---

## Overview

Kubeflow Pipelines (KFP) is the engine behind Vertex AI Pipelines, but testing KFP components is a well-known pain point. Most tutorials show you how to build a pipeline — few show you how to **test it locally before deploying**.

This guide covers the full testing workflow:

- 🧩**KFP Components** — Build a 3-step pipeline (preprocess → train → evaluate)
- ✅**Unit Testing** — Test component logic in isolation with `component.python_func`
- 🔮**Mocking GCP Services** — Simulate BigQuery and GCS without real credentials
- 🐞**Local Execution** — Run the full pipeline locally with `kfp.local.init()` and DockerRunner
- 🔍**Debugging Walkthrough** — Inject a bug, find it locally, fix it, verify
- 🚀**CI/CD** — Add pipeline tests to Cloud Build

No fictional use case needed — just the mechanics, end to end.

Find the code on my GitHub.

---

## 📋The Pipeline

We'll work with a simple 3-component pipeline that's representative of real ML workflows:

```
load_data → train_model → evaluate_model
```

Each component is a `@component`-decorated Python function. Here's the full pipeline:

### 1. Load Data Component

This component simulates loading data from BigQuery (we'll mock the actual client):

```python
from kfp.dsl import component, Output, Dataset, Metrics
from typing import NamedTuple

@component(
    base_image="python:3.10",
    packages_to_install=["google-cloud-bigquery==3.24.0", "pandas==2.0.3"],
)
def load_data_op(
    query: str,
    project_id: str,
    dataset: Output[Dataset],
) -> NamedTuple("Outputs", [("row_count", int)]):
    import pandas as pd
    from google.cloud import bigquery

    client = bigquery.Client(project=project_id)
    df = client.query(query).to_dataframe()
    df.to_csv(dataset.path, index=False)
    return (len(df),)
```

### 2. Train Model Component

A lightweight training step using scikit-learn:

```python
@component(
    base_image="python:3.10",
    packages_to_install=["pandas==2.0.3", "scikit-learn==1.3.0"],
)
def train_model_op(
    dataset: Input[Dataset],
    model_output: Output[Model],
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    import pandas as pd
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(dataset.path)
    X = df.drop("target", axis=1)
    y = df["target"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    import joblib
    joblib.dump(model, model_output.path)
```

### 3. Evaluate Model Component

```python
@component(
    base_image="python:3.10",
    packages_to_install=["pandas==2.0.3", "scikit-learn==1.3.0"],
)
def evaluate_model_op(
    model: Input[Model],
    dataset: Input[Dataset],
    metrics: Output[Metrics],
) -> NamedTuple("Outputs", [("accuracy", float)]):
    import pandas as pd
    import joblib
    from sklearn.metrics import accuracy_score

    df = pd.read_csv(dataset.path)
    X = df.drop("target", axis=1)
    y = df["target"]

    clf = joblib.load(model.path)
    y_pred = clf.predict(X)
    acc = accuracy_score(y, pred)

    metrics.log_metric("accuracy", acc)
    return (acc,)
```

### 4. Pipeline Definition

Tying the components together:

```python
from kfp import dsl

@dsl.pipeline(
    name="ml-pipeline",
    description="End-to-end ML pipeline",
)
def ml_pipeline(
    query: str,
    project_id: str,
    test_size: float = 0.2,
):
    load_task = load_data_op(query=query, project_id=project_id)
    train_task = train_model_op(
        dataset=load_task.outputs["dataset"],
        test_size=test_size,
    )
    eval_task = evaluate_model_op(
        model=train_task.outputs["model_output"],
        dataset=load_task.outputs["dataset"],
    )
```

---

## 🧪Step 2: Unit Testing Components

The `.python_func` attribute gives you access to the raw Python function behind any component. This lets you test component logic without running a container.

### Testing load_data_op

```python
# tests/test_load_data.py
from unittest.mock import patch, MagicMock
from components import load_data_op

@patch("google.cloud.bigquery.Client")
def test_load_data(mock_bq_client, tmp_path):
    # Arrange: mock BigQuery to return a DataFrame
    mock_df = MagicMock()
    mock_df.to_dataframe.return_value = pd.DataFrame({
        "feature1": [1, 2, 3],
        "target": [0, 1, 0],
    })
    mock_bq_client.return_value.query.return_value = mock_df

    dataset_path = str(tmp_path / "data.csv")

    # Act: call the component's underlying function
    func = load_data_op.python_func
    row_count = func(
        query="SELECT * FROM dataset.table",
        project_id="test-project",
        dataset=dataset_path,
    )

    # Assert
    assert row_count == 3
    assert pd.read_csv(dataset_path).shape[0] == 3
```

### Testing train_model_op

```python
# tests/test_train_model.py
from components import train_model_op

def test_train_model(tmp_path):
    # Arrange: create a test CSV
    data_path = tmp_path / "data.csv"
    pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0],
        "feature2": [0.5, 1.5, 2.5, 3.5],
        "target": [0, 0, 1, 1],
    }).to_csv(data_path, index=False)

    model_path = tmp_path / "model.pkl"

    # Act
    func = train_model_op.python_func
    func(
        dataset=data_path,
        model_output=model_path,
        test_size=0.25,
        random_state=42,
    )

    # Assert: model file was created
    import joblib
    model = joblib.load(model_path)
    assert model is not None
    assert hasattr(model, "predict")
```

### Testing evaluate_model_op

```python
# tests/test_evaluate.py
from components import evaluate_model_op

def test_evaluate_model(tmp_path):
    # Arrange
    data_path = tmp_path / "data.csv"
    pd.DataFrame({
        "feature1": [1.0, 2.0, 3.0, 4.0],
        "feature2": [0.5, 1.5, 2.5, 3.5],
        "target": [0, 0, 1, 1],
    }).to_csv(data_path, index=False)

    model_path = tmp_path / "model.pkl"
    import joblib
    from sklearn.linear_model import LogisticRegression
    joblib.dump(LogisticRegression(max_iter=1000).fit(
        [[1.0, 0.5], [2.0, 1.5], [3.0, 2.5], [4.0, 3.5]],
        [0, 0, 1, 1]
    ), model_path)

    # Act
    from unittest.mock import MagicMock
    func = evaluate_model_op.python_func
    accuracy = func(
        dataset=data_path,
        model=model_path,
        metrics=MagicMock(),  # Mock Output[Metrics] artifact
    )

    # Assert
    assert 0.0 <= accuracy <= 1.0
```

**Key insight:** `.python_func` strips away the KFP container wrapper. Inputs become plain values instead of `Input[Dataset]` objects — you pass file paths directly.

---

## 🔮Step 3: Mocking BigQuery and GCS

For components that call GCP services, you need to mock the client libraries. Here's a reusable pattern:

```python
from unittest.mock import patch

@pytest.fixture
def mock_bigquery():
    with patch("google.cloud.bigquery.Client") as mock:
        # Mock query execution
        mock_query_result = MagicMock()
        mock_query_result.to_dataframe.return_value = pd.DataFrame({
            "feature1": [1.0, 2.0, 3.0],
            "target": [0, 1, 0],
        })
        mock.return_value.query.return_value = mock_query_result
        yield mock

@pytest.fixture
def mock_gcs():
    with patch("google.cloud.storage.Client") as mock:
        mock_bucket = MagicMock()
        mock_blob = MagicMock()
        mock_bucket.blob.return_value = mock_blob
        mock.return_value.bucket.return_value = mock_bucket
        yield mock
```

**Why mocking matters for CI/CD:**

- No GCP project or service account required
- Tests run in seconds, not minutes
- Error conditions are easy to simulate (network failure, empty results, quota exceeded)

---

## 🐞Step 4: Running the Pipeline Locally with kfp.local.init()

KFP v2 introduced `kfp.local.init()` which lets you run a full pipeline locally using Docker or a Python subprocess.

### Setup

```bash
pip install kfp kfp-local
```

Docker is optional — you can run in `subprocess` mode instead of Docker mode.

### Local Execution

```python
import kfp

# Initialize local runner
kfp.local.init()

# Run the pipeline locally
result = kfp.local.run_pipeline(
    pipeline_func=ml_pipeline,
    params={
        "query": "SELECT * FROM dataset.table",
        "project_id": "test-project",
    },
)

print(f"Pipeline run completed: {result.status}")
```

This executes each component as a separate Python subprocess. Artifacts are written to a local directory.

### Using DockerRunner

For a more realistic test (each component runs in its own container):

```python
from kfp.local import DockerRunner

kfp.local.init(runner=DockerRunner())

result = kfp.local.run_pipeline(
    pipeline_func=ml_pipeline,
    params={
        "query": "SELECT * FROM dataset.table",
        "project_id": "test-project",
    },
)
```

Docker mode builds a container for each component using the `base_image` and `packages_to_install` from the decorator. This catches dependency issues early.

### Inspecting Results

```python
print(f"Status: {result.status}")
for task_name, task in result.tasks.items():
    print(f"  {task_name}: {task.status}")
    if task.outputs:
        for name, value in task.outputs.items():
            print(f"    {name}: {value}")
```

---

## 🔍Step 5: Debugging Walkthrough

Let's debug a real failure. We'll intentionally break the pipeline and walk through the fix.

### The Bug

The evaluate_model_op has a typo: `accuracy_score(y, pred)` should be `accuracy_score(y, y_pred)`:

```python
# Buggy line in evaluate_model_op:
acc = accuracy_score(y, pred)  # NameError: 'pred' is not defined
```

### Step 5.1: Run the Pipeline Locally

```bash
python run_pipeline.py
```

Output:

```
Traceback (most recent call last):
  ...
  File "components.py", line 52, in evaluate_model_op
    acc = accuracy_score(y, pred)
NameError: name 'pred' is not defined
```

### Step 5.2: Isolate with Unit Test

Run the unit test for evaluate_model_op:

```bash
pytest tests/test_evaluate.py -v
```

Output:

```
FAILED test_evaluate.py::test_evaluate_model - NameError: name 'pred' is not defined
```

### Step 5.3: Add Debug Logging

Temporarily add logging to understand the failure:

```python
import logging
logging.info(f"DEBUG: y has {len(y)} samples")
logging.info(f"DEBUG: y_pred has {len(y_pred)} samples")
```

Re-run:

```
DEBUG: y has 4 samples
DEBUG: y_pred has 4 samples
```

The data is fine — it's a variable name mismatch.

### Step 5.4: Fix and Verify

```python
# Fixed:
acc = accuracy_score(y, y_pred)
```

Run the unit test again:

```bash
pytest tests/test_evaluate.py -v
```

Output:

```
PASSED test_evaluate.py::test_evaluate_model
```

### Step 5.5: Re-run Pipeline Locally

```bash
python run_pipeline.py
```

Output:

```
Pipeline run completed: SUCCESS
load_task: SUCCESS
train_task: SUCCESS
eval_task: SUCCESS
```

### Step 5.6: Verify Artifacts

```bash
ls -la _local_run/artifacts/
```

Check that model and metrics files were created.

---

## 🚀Step 6: CI/CD with Cloud Build

Add pipeline tests to your Cloud Build configuration:

```yaml
# cloudbuild.yaml
steps:
  - name: python:3.10
    entrypoint: pip
    args: ["install", "-r", "requirements.txt"]

  - name: python:3.10
    entrypoint: pip
    args: ["install", "kfp", "pytest"]

  - name: python:3.10
    entrypoint: python
    args: ["-m", "pytest", "tests/", "-v"]

  - name: python:3.10
    entrypoint: python
    args: ["run_pipeline.py"]

  - name: gcr.io/cloud-builders/gcloud
    entrypoint: python
    args: ["-m", "kfp", "compiler", "compile",
           "--pipeline-file", "pipeline.py",
           "--output", "pipeline.yaml"]

  - name: gcr.io/cloud-builders/gcloud
    entrypoint: python
    args: ["-m", "kfp", "registry", "upload",
           "--pipeline-file", "pipeline.yaml",
           "--pipeline-name", "ml-pipeline"]
```

### Running Tests in Cloud Build

```bash
gcloud builds submit --config cloudbuild.yaml
```

Cloud Build runs:

1. Install dependencies
2. Run unit tests (`pytest`)
3. Run local pipeline (validates execution)
4. Compile the pipeline (validates KFP syntax)
5. Upload to Artifact Registry

### GitHub Actions Alternative

```yaml
# .github/workflows/test-pipeline.yml
name: Test KFP Pipeline
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.10"
      - run: pip install kfp pytest scikit-learn pandas google-cloud-bigquery
      - run: pytest tests/ -v
      - run: python run_pipeline.py
```

---

## ❗Common Issues

### Issue: "pipeline_func is not compiled"

**Error:** `AttributeError: 'function' object has no attribute 'pipeline_func'`

**Solution:** Use `kfp.local.run_pipeline(pipeline_func=ml_pipeline, ...)` with `kfp.local.init()` called first, not the compiler API.

### Issue: "Docker not available" in CI

**Error:** `Docker not found. Install Docker or use subprocess runner.`

**Solution:** Use the default subprocess runner in CI:

```python
kfp.local.init()  # default subprocess runner, no Docker needed
```

### Issue: "Mocked BQ client not being used"

**Error:** Tests hit real BigQuery despite mocks.

**Solution:** Apply the patch at the correct import location:

```python
# Patch where it's used, not where it's defined
@patch("components.bigquery.Client")  # Patch the import in your module
def test_load_data(mock_bq):
    ...
```

### Issue: "Output[Dataset] returns None in unit tests"

**Error:** When calling `.python_func` directly, `Output[Dataset]` parameters receive `None` instead of a path.

**Solution:** Pass a file path string directly instead of an `Output[Dataset]`:

```python
func = load_data_op.python_func
func(query=..., project_id=..., dataset="/tmp/test_data.csv")  # pass string path
```

---

## 💡Key Takeaways

- **`component.python_func`** is your primary unit testing tool — it exposes the raw function
- **`kfp.local.init()`** runs the full pipeline locally without Vertex AI
- **Mock GCP clients** for fast, credential-free CI tests
- **Debug in layers**: unit test → local run → compile → deploy
- **Run tests in CI** (Cloud Build or GitHub Actions) to catch regressions early

Let's connect on [LinkedIn](https://www.linkedin.com/in/pallav-anand-025526109/)!

**Tags:**
- Kubeflow Pipelines
- Vertex AI
- MLOps
- CI/CD
- Google Cloud Platform
- Machine Learning
