from datetime import datetime

from endpoint import AIEndpoint
from monitoring import Monitoring
from train import Train


def pipeline():
    timestamp_str = datetime.strftime(datetime.now(), '%y%m%d_%H%M%S')
    train = Train()
    train.create_custom_job_with_experiment_autologging_sample(display_name=f"iris_{timestamp_str}",
                                                               tune_hyperparameters=False)

    ai_endpoint = AIEndpoint()
    endpoint = ai_endpoint.deploy_endpoint(endpoint_name='isis-endpoint')

    monitoring = Monitoring(endpoint)
    monitoring.config_monitoring(target='iris')


if __name__ == '__main__':
    pipeline()
