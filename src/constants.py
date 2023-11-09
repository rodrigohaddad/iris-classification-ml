LOG_SAMPLE_RATE = 0.8
MONITOR_INTERVAL = 1
DEFAULT_THRESHOLD_VALUE = 0.001
THRESHOLDS = {
    "sepal length": DEFAULT_THRESHOLD_VALUE,
    "petal width": DEFAULT_THRESHOLD_VALUE,
}


class TrainingArgs:
    replica_count = 1
    epochs = 15
    batch_size = 20
    lr = 0.001
