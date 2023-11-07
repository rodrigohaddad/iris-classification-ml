import os

from dotenv import load_dotenv
from google.cloud import aiplatform
from google.cloud.aiplatform import Endpoint
from google.cloud.aiplatform.explain.metadata.tf.v2 import \
    saved_model_metadata_builder

load_dotenv()

endpoint = "projects/900832571968/locations/southamerica-east1/endpoints/3763162108947070976"


class AIEndpoint:
    def __init__(self, project: str, location: str):
        self.project = project
        self.location = location
        aiplatform.init(project=self.project, location=self.location)

        self.explain_params, self.explain_meta = self.get_explain_config()

    def get_explain_config(self):
        params = {"sampled_shapley_attribution": {"path_count": 10}}
        explain_params = aiplatform.explain.ExplanationParameters(params)

        builder = saved_model_metadata_builder.SavedModelMetadataBuilder(
            model_path=f"{os.getenv('MODEL_OUT_DIR')}/savedmodel",
            outputs_to_explain=['fare']
        )
        explain_meta = builder.get_metadata_protobuf()

        return explain_params, explain_meta

    def deploy_endpoint(self, endpoint_name) -> Endpoint:
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
            max_replica_count=1,
            explanation_parameters=self.explain_params,
            explanation_metadata=self.explain_meta,

        )
        return endpoint


if __name__ == '__main__':
    endpoint = AIEndpoint(project=os.getenv('PROJECT'),
                          location=os.getenv('REGION'))
    endpoint.deploy_endpoint(endpoint_name='taxi-fare-endpoint')
