import os

from dotenv import load_dotenv
from google.cloud import aiplatform
from google.cloud.aiplatform import Endpoint
from google.cloud.aiplatform.explain.metadata.tf.v2 import \
    saved_model_metadata_builder

load_dotenv()


class AIEndpoint:
    def __init__(self):
        self.project = os.getenv('PROJECT')
        self.location = os.getenv('REGION')
        aiplatform.init(project=self.project, location=self.location)

        # self.explain_params, self.explain_meta = self.get_explain_config()

    @staticmethod
    def get_explain_config():
        params = {"sampled_shapley_attribution": {"path_count": 10}}
        explain_params = aiplatform.explain.ExplanationParameters(params)

        # builder = saved_model_metadata_builder.SavedModelMetadataBuilder(
        #     model_path=f"{os.getenv('MODEL_OUT_DIR')}/savedmodel",
        #     outputs_to_explain=['class']
        # )

        builder = saved_model_metadata_builder.SavedModelMetadataBuilder(
            model_path=f"C:\\Users\\rodri\\Documents\\projects\\iris-classification-ml\\ds\\local_data",
            outputs_to_explain=['class']
        )
        explain_meta = builder.get_metadata_protobuf()

        return explain_params, explain_meta

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
