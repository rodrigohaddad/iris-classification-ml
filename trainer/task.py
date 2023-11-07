import argparse
from trainer import model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch_size",
        help="Batch size for training steps",
        type=int,
        default=32,
    )
    parser.add_argument(
        "--eval_data_path",
        help="GCS location pattern of eval files",
        required=True,
    )
    parser.add_argument(
        "--lr", help="learning rate for optimizer", type=float, default=0.001
    )
    parser.add_argument(
        "--output_dir",
        help="GCS location to write checkpoints and export models",
        required=True,
    )
    parser.add_argument(
        "--train_data_path",
        help="GCS location pattern of train files containing eval URLs",
        required=True,
    )
    args = parser.parse_args()
    hparams = args.__dict__

    model = model.Model(hparams)
    model.train_and_evaluate()
