import os
from datetime import datetime

from dotenv import load_dotenv
from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt

from constants import TrainingArgs

load_dotenv()


class Train:
    def __init__(self, project, location):
        self.project = project
        self.location = location
        aiplatform.init(project=self.project, location=self.location)

    def create_custom_job_with_experiment_autologging_sample(
            self,
            display_name: str,
            tune_hyperparameters: bool = False,
            # experiment: str,
            # experiment_run: Optional[str] = None,
    ) -> None:
        custom_job = {
            "display_name": display_name,
            "staging_bucket": os.getenv('STAGING_DIR'),
            "worker_pool_specs": [
                {
                    "machine_spec": {
                        "machine_type": os.getenv('MACHINE_TYPE'),
                    },
                    "replica_count": TrainingArgs.replica_count,
                    "python_package_spec": {
                        "executor_image_uri": os.getenv("PYTHON_PACKAGE_EXECUTOR_IMAGE_URI"),
                        "package_uris": [os.getenv("PYTHON_PACKAGE_URIS")],
                        "python_module": os.getenv("PYTHON_MODULE"),
                        "args": [
                            f"--eval_data_path={os.getenv('EVAL_DIR')}",
                            f"--train_data_path={os.getenv('TRAIN_DIR')}",
                            f"--bucket_name={os.getenv('BUCKET')}",
                            f"--output_dir={os.getenv('OUT_DIR')}",
                            f"--epochs={TrainingArgs.epochs}",
                            f"--lr={TrainingArgs.lr}",
                        ],
                    },
                }
            ],
        }

        job = aiplatform.CustomJob(**custom_job)

        if tune_hyperparameters:
            job = self.add_hyperparameter_tuning(display_name, job)
        job.submit()

    @staticmethod
    def add_hyperparameter_tuning(display_name, job):
        hpt_job = aiplatform.HyperparameterTuningJob(
            display_name=display_name,
            custom_job=job,
            metric_spec={'loss': 'minimize'},
            parameter_spec={
                'lr': hpt.DoubleParameterSpec(min=0.001, max=0.1, scale='log'),
            },
            max_trial_count=20,
            parallel_trial_count=1,
        )

        return hpt_job


if __name__ == '__main__':
    timestamp_str = datetime.strftime(datetime.now(), '%y%m%d_%H%M%S')
    train = Train(project=os.getenv('PROJECT'),
                  location=os.getenv('REGION'))
    train.create_custom_job_with_experiment_autologging_sample(display_name=f"iris_{timestamp_str}",
                                                               tune_hyperparameters=False)
