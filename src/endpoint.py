import os

from dotenv import load_dotenv
from google.cloud import aiplatform
from google.cloud.aiplatform import Endpoint

load_dotenv()


class AIEndpoint:
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
