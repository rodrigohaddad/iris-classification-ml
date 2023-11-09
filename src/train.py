import os

from dotenv import load_dotenv
from google.cloud import aiplatform
from google.cloud.aiplatform import hyperparameter_tuning as hpt

from constants import TrainingArgs

load_dotenv()


class Train:
    def create_custom_job(self, display_name: str, tune_hyperparameters: bool = False) -> None:
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
                            f"--output_dir={os.getenv('OUT_DIR')}",
                            f"--epochs={TrainingArgs.epochs}",
                            f"--batch_size={TrainingArgs.epochs}",
                            f"--lr={TrainingArgs.lr}",
                        ],
                    },
                }
            ],
        }

        job = aiplatform.CustomJob(**custom_job)

        if tune_hyperparameters:
            job = self.add_hyperparameter_tuning(display_name, job)
        job.run()

    @staticmethod
    def add_hyperparameter_tuning(display_name: str, job) -> aiplatform.HyperparameterTuningJob:
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
