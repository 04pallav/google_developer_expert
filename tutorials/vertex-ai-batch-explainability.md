# ☁️Vertex AI Batch Explainability: Making ML Models Transparent🔍📊

**By jana_om**

---

## Overview

This GCP Machine Learning project focuses on implementing **batch explainability** for ML models using Vertex AI. The solution is designed to generate explanations for model predictions, leveraging various Google Cloud Platform (GCP) services:

- 🗃️**Cloud Storage** — Store model artifacts and batch input data
- 🤖**Vertex AI Prediction** — Deploy models with explainability enabled
- 🔍**Vertex Explainable AI** — Generate SHAP-based explanations for predictions
- 📊**BigQuery** — Store and analyze explanation results
- 📈**Looker Studio** — Visualize feature attributions

These technologies work together to efficiently generate, store, and analyze model explanations.

Find the code on my GitHub.

---

## 🎯 What is Batch Explainability?

Batch explainability lets you understand **why a machine learning model made predictions** for **many examples at once** — not just one at a time.

**Use it when:**
- ✅ You have 1,000+ predictions to explain (e.g., loan applications, user recommendations)
- ✅ You need explanations in bulk for reporting, auditing, or compliance
- ✅ You want to save cost compared to real-time (online) explainability

---

## 🗃️Step 1: Prepare Your Model and Data

### Create a Cloud Storage bucket

```bash
BUCKET_URI="gs://$PROJECT_ID-vertex-explainability"
gcloud storage buckets create --location=$REGION $BUCKET_URI
```

### Upload your model artifacts

Your model should be in a format Vertex AI supports (SavedModel, XGBoost, scikit-learn, etc.):

```bash
# Upload model to Cloud Storage
gcloud storage cp model.joblib $BUCKET_URI/models/
```

### Prepare batch input data

Create a JSON file with instances to explain:

**instances.json:**
```json
{
  "instances": [
    {
      "sepal_length": 4.8,
      "sepal_width": 3.0,
      "petal_length": 1.4,
      "petal_width": 0.3
    },
    {
      "sepal_length": 6.2,
      "sepal_width": 3.4,
      "petal_length": 5.4,
      "petal_width": 2.3
    }
  ]
}
```

Upload to Cloud Storage:

```bash
gcloud storage cp instances.json $BUCKET_URI/batch-input/
```

---

## 🐝Step 2: Configure Explainability

### Choose your explanation method

Vertex AI supports multiple explanation methods:

| Method | Best For | Description |
|---|---|---|
| **Shapley** | Tabular data | Game theory-based feature attribution |
| **Integrated Gradients** | Images, text | Gradient-based attribution |
| **XRAI** | Images | Region-based highlighting |

For this tutorial, we'll use **Shapley** (recommended for tabular data):

```python
from google.cloud import aiplatform

# Initialize Vertex AI SDK
aiplatform.init(project=PROJECT_ID, location=REGION)

# Configure Shapley explanation parameters
explanation_parameters = aiplatform.explain.ExplanationParameters({
    "sampled_shapley_attribution": {"path_count": 10}
})
```

### Configure explanation metadata

Tell Vertex AI which features are inputs and which outputs to explain:

```python
explanation_metadata = aiplatform.explain.ExplanationMetadata(
    inputs={
        "sepal_length": {},
        "sepal_width": {},
        "petal_length": {},
        "petal_width": {},
    },
    outputs={"probability": {}},
)
```

❗ **Important:** Input feature names must match your model's expected input format exactly.

---

## 🚀Step 3: Deploy Model with Explainability

### Upload the model to Vertex AI

```python
model = aiplatform.Model.upload(
    display_name="iris-classifier-explainable",
    artifact_uri=f"{BUCKET_URI}/models/",
    serving_container_image_uri="us-docker.pkg.dev/vertex-ai/prediction/sklearn-cpu.1-0:latest",
    explanation_parameters=explanation_parameters,
    explanation_metadata=explanation_metadata,
)
```

### Deploy the model

```python
endpoint = model.deploy(machine_type="n1-standard-4")
```

Press enter or click to view image in full size

*[Image: Vertex AI Model Deployment Console showing explainability configuration]*

---

## 📡Step 4: Run Batch Predictions with Explanations

### Using Python SDK

```python
instances = [
    {"sepal_length": 4.8, "sepal_width": 3.0, "petal_length": 1.4, "petal_width": 0.3},
    {"sepal_length": 6.2, "sepal_width": 3.4, "petal_length": 5.4, "petal_width": 2.3},
]

# Get predictions
predictions = endpoint.predict(instances=instances)
print("Predictions:", predictions.predictions)

# Get explanations
explanations = endpoint.explain(instances=instances)
print("Explanations:", explanations.explanations)
```

### Using gcloud CLI

```bash
# Get predictions
gcloud beta ai endpoints predict $ENDPOINT_ID \
  --region=$REGION \
  --json-request=instances.json

# Get explanations
gcloud beta ai endpoints explain $ENDPOINT_ID \
  --region=$REGION \
  --json-request=instances.json
```

Press enter or click to view image in full size

*[Image: Terminal showing gcloud explain command output with SHAP values]*

### Using REST API

```bash
# Get explanations via REST
curl \
  -H "Authorization: Bearer $(gcloud auth print-access-token)" \
  -H "Content-Type: application/json" \
  -d @instances.json \
  "https://$REGION-aiplatform.googleapis.com/v1/projects/$PROJECT_ID/locations/$REGION/endpoints/$ENDPOINT_ID:explain"
```

---

## 📊Step 5: Analyze and Visualize Results

### Understanding the output

Explanation response includes:

```json
{
  "explanations": [
    {
      "attributions": [
        {
          "featureAttributions": {
            "sepal_length": 0.23,
            "sepal_width": -0.15,
            "petal_length": 0.08,
            "petal_width": 0.04
          }
        }
      ]
    }
  ]
}
```

- **Positive score** → Feature increased the prediction probability
- **Negative score** → Feature decreased the prediction probability
- **Magnitude** → How much the feature influenced the prediction

### Visualize feature attributions

```python
import matplotlib.pyplot as plt
import pandas as pd

# Extract attributions from response
attributions = explanations.explanations[0].attributions
feature_scores = attributions[0].feature_attributions

# Create DataFrame
df = pd.DataFrame({
    'Feature': list(feature_scores.keys()),
    'SHAP Value': list(feature_scores.values())
})

# Plot
plt.figure(figsize=(10, 6))
plt.barh(df['Feature'], df['SHAP Value'])
plt.title('Feature Importance for Prediction')
plt.xlabel('SHAP Value')
plt.tight_layout()
plt.show()
```

Press enter or click to view image in full size

*[Image: Horizontal bar chart showing feature importance with SHAP values]*

### Store results in BigQuery

```python
from google.cloud import bigquery

client = bigquery.Client(project=PROJECT_ID)

# Create table
table_id = f"{PROJECT_ID}.ml_explainability.explanations"
job_config = bigquery.LoadJobConfig(
    source_format=bigquery.SourceFormat.NEWLINE_DELIMITED_JSON,
)

# Load explanations
job = client.load_table_from_json(
    explanations.explanations,
    table_id,
    job_config=job_config,
)
job.result()
```

---

## 🌠Step 6: Build a Dashboard (Optional)

Connect Looker Studio to BigQuery:

1. Go to [lookerstudio.google.com](https://lookerstudio.google.com)
2. Create new report → Select BigQuery connector
3. Choose your `ml_explainability.explanations` table
4. Build charts:
   - Feature importance over time
   - Prediction confidence distribution
   - Anomaly detection (unexpected feature attributions)

Press enter or click to view image in full size

*[Image: Looker Studio dashboard showing ML model explainability metrics]*

---

## ❗Common Issues

### Issue: "Feature name mismatch"

**Error:** `Invalid argument: Feature name 'age' not found in explanation metadata`

**Solution:** Ensure input feature names exactly match the metadata configuration:

```python
explanation_metadata = aiplatform.explain.ExplanationMetadata(
    inputs={
        "age": {},  # Must match input JSON exactly
        "income": {},
        "credit_score": {},
    }
)
```

### Issue: "Explanation timeout for large batches"

**Error:** `Deadline exceeded for batch of 10,000 instances`

**Solution:** Split into smaller batches:

```python
batch_size = 1000
for i in range(0, len(instances), batch_size):
    batch = instances[i:i+batch_size]
    explanation = endpoint.explain(instances=batch)
```

---

Let's connect on [LinkedIn](https://www.linkedin.com/in/jana-polianskaja/)! 💬😊

Great news! Apache Beam reposted my project on Linkedin! 🤩🥳

**Tags:**
- Vertex AI
- Explainable AI
- Machine Learning
- Google Cloud Platform
- MLOps
