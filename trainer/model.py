import logging
import os
import numpy as np
import tensorflow as tf
from keras import callbacks, models, layers

from iris.trainer.file_manipulation import load_df

logging.info(tf.version.VERSION)


class Model:
    def __init__(self, hparams: dict):
        self.lr = hparams['lr']
        self.epochs = hparams['epochs']
        self.batch_size = hparams['batch_size']

        self.output_dir = hparams['output_dir']
        self.eval_data_path = hparams['eval_data_path']
        self.train_data_path = hparams['train_data_path']

    @staticmethod
    def transform(trainds, evalds):
        _, y_train = np.unique(trainds['class'], return_inverse=True)
        _, y_val = np.unique(evalds['class'], return_inverse=True)
        y_train = tf.keras.utils.to_categorical(y_train, num_classes=3)
        y_val = tf.keras.utils.to_categorical(y_val, num_classes=3)

        x_train = trainds.drop('class', axis=1)
        x_val = evalds.drop('class', axis=1)
        return (x_train, x_val), (y_train, y_val)

    def build_nn(self) -> models:
        nn = tf.keras.Sequential([
            layers.Input(shape=(4,)),
            layers.Dense(64, activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(3, activation='softmax')
        ])

        lr_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        nn.compile(optimizer=lr_optimizer,
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
        return nn

    def train_and_evaluate(self):
        model_export_path = os.path.join(self.output_dir, "savedmodel")
        checkpoint_path = os.path.join(self.output_dir, "checkpoints")
        tensorboard_path = os.path.join(self.output_dir, "tensorboard")

        if tf.io.gfile.exists(self.output_dir):
            tf.io.gfile.rmtree(self.output_dir)

        trainds = load_df(self.train_data_path)
        evalds = load_df(self.eval_data_path)

        x, y = self.transform(trainds, evalds)

        nn = self.build_nn()
        logging.info(nn.summary())

        checkpoint_cb = callbacks.ModelCheckpoint(
            checkpoint_path, save_weights_only=True, verbose=1
        )
        tensorboard_cb = callbacks.TensorBoard(tensorboard_path, histogram_freq=1)

        history = nn.fit(
            x=x[0],
            y=y[0],
            validation_data=(x[1], y[1]),
            epochs=self.epochs,
            verbose=2,
            callbacks=[checkpoint_cb, tensorboard_cb],
        )

        nn.save(model_export_path)
        return history


if __name__ == "__main__":
    model = Model({'lr': 0.001,
                   'epochs': 15,
                   'batch_size': 10,
                   'num_examples_to_train_on': 100,
                   'output_dir': 'C:\\Users\\rodri\\Documents\\projects\\iris-classification-ml\\ds\\local_data\\output',
                   'eval_data_path': 'C:\\Users\\rodri\\Documents\\projects\\iris-classification-ml\\ds\\local_data\\eval\\eval.csv',
                   'train_data_path': 'C:\\Users\\rodri\\Documents\\projects\\iris-classification-ml\\ds\\local_data\\train\\train.csv'
                   },
                   )
    model.train_and_evaluate()
