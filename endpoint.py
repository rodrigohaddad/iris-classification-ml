import os

from dotenv import load_dotenv
from google.cloud import aiplatform
from google.cloud.aiplatform import Endpoint

load_dotenv()

MODEL = 'projects/1079154697342/locations/southamerica-east1/models/6917054038617882624/operations/3611575739360477184'
ENDPOINT = 'projects/1079154697342/locations/southamerica-east1/endpoints/7775447164469116928/operations/792322372626546688'

class AIEndpoint:
    def __init__(self, project: str, location: str):
        self.project = project
        self.location = location
        aiplatform.init(project=self.project, location=self.location)

    @staticmethod
    def deploy_endpoint(endpoint_name: str) -> Endpoint:
        try:
            model = aiplatform.Model.list()[0]
        except Exception as e:
            model = aiplatform.Model.upload(artifact_uri=f"{os.getenv('MODEL_OUT_DIR')}/savedmodel",
                                            serving_container_image_uri=os.getenv("TF_SERVING_IMG"))

        endpoint = aiplatform.Endpoint(endpoint_name=endpoint_name)
        endpoint = endpoint.create()
        endpoint.deploy(
            model=model,
            traffic_percentage=100,
            machine_type=os.getenv('MACHINE_TYPE_SERVING'),
            min_replica_count=1,
            max_replica_count=1
        )

        return endpoint


if __name__ == '__main__':
    ai_endpoint = AIEndpoint(project=os.getenv('PROJECT'),
                             location=os.getenv('REGION'))
    ai_endpoint.deploy_endpoint(endpoint_name='isis-endpoint')

