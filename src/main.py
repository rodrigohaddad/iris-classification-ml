import os
import functions_framework
from datetime import datetime
from endpoint import AIEndpoint
from monitoring import Monitoring
from train import Train
from google.cloud import aiplatform


@functions_framework.cloud_event
def pipeline(cloud_event):
    project = os.getenv('PROJECT')
    location = os.getenv('REGION')
    aiplatform.init(project=project, location=location)

    timestamp_str = datetime.strftime(datetime.now(), '%y%m%d_%H%M%S')
    train = Train()
    train.create_custom_job(display_name=f"iris_{timestamp_str}",
                            tune_hyperparameters=False)

    ai_endpoint = AIEndpoint()
    endpoint = ai_endpoint.deploy_endpoint(endpoint_name='isis-endpoint')

    monitoring = Monitoring(endpoint)
    monitoring.config_monitoring(target='iris')


if __name__ == '__main__':
    pipeline("")
