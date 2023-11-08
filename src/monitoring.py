import os

from dotenv import load_dotenv
from google.cloud.aiplatform import model_monitoring
import google.cloud.aiplatform as aiplatform

load_dotenv()

# Sampling rate (optional, default=.8)
LOG_SAMPLE_RATE = 0.8

# Monitoring Interval in hours (optional, default=1).
MONITOR_INTERVAL = 1

DEFAULT_THRESHOLD_VALUE = 0.001

THRESHOLDS = {
    "passenger_count": DEFAULT_THRESHOLD_VALUE,
    "dropoff_latitude": DEFAULT_THRESHOLD_VALUE,
}


class Monitoring:
    def __init__(self):
        self.project = os.getenv('PROJECT')
        self.location = os.getenv('REGION')
        aiplatform.init(project=self.project, location=self.location)
        endpoint_name = "projects/900832571968/locations/southamerica-east1/endpoints/3763162108947070976"
        self.endpoint = aiplatform.Endpoint(endpoint_name=endpoint_name)

    def config_monitoring(self, target):
        skew_config = model_monitoring.SkewDetectionConfig(
            data_source=f"{os.getenv('OUT_DIR')}/taxi-train-000000000000.csv",
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


if __name__ == '__main__':
    # mdm = aiplatform.ModelDeploymentMonitoringJob('7865908384132759552', os.getenv('PROJECT'), os.getenv('REGION'))
    # mdm.delete()
    monitoring = Monitoring()
    monitoring.config_monitoring(target='iris')
