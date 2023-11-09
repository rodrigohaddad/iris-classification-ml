import os

from dotenv import load_dotenv
from google.cloud.aiplatform import model_monitoring
import google.cloud.aiplatform as aiplatform

from constants import THRESHOLDS, LOG_SAMPLE_RATE, MONITOR_INTERVAL

load_dotenv()


class Monitoring:
    def __init__(self, endpoint=None):
        self.project = os.getenv('PROJECT')
        self.location = os.getenv('REGION')
        aiplatform.init(project=self.project, location=self.location)
        self.endpoint = endpoint
        if not endpoint:
            endpoint_name = os.getenv('ENDPOINT')
            self.endpoint = aiplatform.Endpoint(endpoint_name=endpoint_name)

    def config_monitoring(self, target: str):
        skew_config = model_monitoring.SkewDetectionConfig(
            data_source=f"{os.getenv('TRAIN_DIR')}",
            skew_thresholds=THRESHOLDS,
            attribute_skew_thresholds=THRESHOLDS,
            target_field=target,
        )

        drift_config = model_monitoring.DriftDetectionConfig(
            drift_thresholds=THRESHOLDS,
            attribute_drift_thresholds=THRESHOLDS,
        )

        explanation_config = model_monitoring.ExplanationConfig()
        objective_config = model_monitoring.ObjectiveConfig(
            skew_detection_config=skew_config,
            drift_detection_config=drift_config,
            # explanation_config=explanation_config
        )

        # Create sampling configuration
        random_sampling = model_monitoring.RandomSampleConfig(sample_rate=LOG_SAMPLE_RATE)

        # Create schedule configuration
        schedule_config = model_monitoring.ScheduleConfig(monitor_interval=MONITOR_INTERVAL)

        # Create alerting configuration.
        emails = [os.getenv('USER_EMAIL')]
        alerting_config = model_monitoring.EmailAlertConfig(
            user_emails=emails, enable_logging=True
        )

        # Create the monitoring job.
        job = aiplatform.ModelDeploymentMonitoringJob.create(
            display_name='monitoring',
            logging_sampling_strategy=random_sampling,
            schedule_config=schedule_config,
            alert_config=alerting_config,
            objective_configs=objective_config,
            project=self.project,
            location=self.location,
            endpoint=self.endpoint,
        )

        return job
