# Iris Classification ML

Iris Classification ML is a machine learning pipeline built on 
Google Cloud Platform. The pipeline is mostly build using Vertex AI
SDK, from training to deployment.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Tests](#tests)
- [Configuration](#configuration)
- [Usage](#usage)

## Requirements

List the prerequisites and dependencies required to use your service, such as:

- Python 3.9+
- Google Cloud SDK (gcloud)
- A Google Cloud project

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/rodrigohaddad/iris-classification-ml.git
   ```

2. Change to the project directory:

   ```bash
   cd iris-classification-ml
   ```
   
3. Create a virtual environment:
    ```bash
   virtualenv venv
   ```
   
4. Activate the virtual environment (on Windows, use venv\Scripts\activate):
    ```bash
   source venv/bin/activate
   ```

5. Install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```
   
6. Configure yor `.env` by using `env` as reference.

## Tests
Execute unit tests:
   ```bash
   python -m unittest
   ```

## Configuration

Provide details on how to configure and authenticate your service, including environment variables, configuration files, and authentication tokens.

### Environment Variables

- `GOOGLE_APPLICATION_CREDENTIALS`: Path to your Google Cloud service account key JSON file.

### Configuration File

You can create a configuration file (`config.yaml`, `config.json`, etc.) for custom settings.

```yaml
# Example config.yaml
project_id: your-project-id
region: us-central1
```

### Authentication

Ensure you are authenticated with the Google Cloud SDK using `gcloud auth application-default login`.

## Usage

For a complete pipeline execution, simply run:

```bash
python main.py
```

By executing `main.py` you will be running the following steps using the Vertex AI pipelines and SDK.

From which:
* **Train**: uses model code as Docker image or library to execute training 
steps with entry hyperparameters or with hyperparameter tuning in a remote cloud machine.
* **Deploy endpoint**: deploy your previously trained model to an online endpoint. Configure
replicas according to traffic.
* **Monitoring**: create a job to monitor endpoint. Supports skew and drift 
monitors. It is also able to provide explanations.

## Automatic pipeline deployment

Commits to master trigger test and deployment of the pipeline contained in `src` to a Cloud Function.

![img3.png](imgs/img3.png)

Cloud Scheduler is set up by using cron notation `* * * * *` to send messages to Pub/Sub and trigger the ML pipeline exeuction:

![img2.png](imgs/img2.png)

Then, the pipelined is triggered by the message and exeuction starts.

![img.png](imgs/img.png)



## Inference results

Currently, there are two ways of applying the model to your test dataset. Load the model and apply input to the trained model:
```python
import tensorflow as tf
import numpy as np

model = tf.saved_model.load(PATH_SAVED_MODEL)
r = model(
        np.array([
    [5.1, 3.5, 1.4, 0.2]
], dtype=np.float32)
)
```

Or, use `gcloud` to send internal GCP `json` requests to Vertex AI endpoint:
```bash
%%bash
gcloud ai endpoints predict $ENDPOINT_RESOURCENAME \
    --region=$REGION \
    --json-request=test_req.json
```

