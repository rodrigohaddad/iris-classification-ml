# Iris Classification ML

Iris Classification ML is a machine learning pipeline built on 
Google Cloud Platform. The pipeline is mostly build using Vertex AI
SDK, from training to deployment.

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Contributing](#contributing)
- [License](#license)

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
   
6. Configure yor `.env`.


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

By executing `main.py` you will be running the following steps using the Vertex AI pipelines and SDK:

From which:
* **Train**: uses model code as Docker image or library to execute training 
steps with entry hyperparameters or with hyperparameter tuning in a remote cloud machine.
* **Deploy endpoint**: deploy your previously trained model to an online endpoint. Configure
replicas according to traffic.
* **Monitoring**: create a job to monitor endpoint. Supports skew and drift 
monitors. It is also able to provide explanations.

## Automatic execution of pipeline

For periodic train and deployment of the inference endpoint, deploy your `src` pipeline code to a Cloud Function and set up a trigger by a Pub/Sub.

![img.png](imgs/img.png)

Configure a Cloud Scheduler using the cron notation `* * * * *`:

![img.png](imgs/img2.png)

