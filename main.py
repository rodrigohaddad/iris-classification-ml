from datetime import datetime

from src.endpoint import AIEndpoint
from src.monitoring import Monitoring
from src.train import Train


def pipeline():
    timestamp_str = datetime.strftime(datetime.now(), '%y%m%d_%H%M%S')
    train = Train()
    train.create_custom_job(display_name=f"iris_{timestamp_str}",
                            tune_hyperparameters=False)

    ai_endpoint = AIEndpoint()
    endpoint = ai_endpoint.deploy_endpoint(endpoint_name='isis-endpoint')

    monitoring = Monitoring(endpoint)
    monitoring.config_monitoring(target='iris')


