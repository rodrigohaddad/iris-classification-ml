import logging
import os
from typing import Tuple

import numpy as np
import tensorflow as tf
import pandas as pd
from keras import callbacks, layers
from keras.src.callbacks import History

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
    def normalize_column(column: pd.Series) -> pd.Series:
        min_val = column.min()
        max_val = column.max()
        return (column - min_val) / (max_val - min_val)

    def transform(self, trainds: pd.DataFrame,
                  evalds: pd.DataFrame) -> (Tuple, Tuple):
        classes, y_train = np.unique(trainds['class'], return_inverse=True)
        mapping_dict = {string: i for i, string in enumerate(sorted(classes))}
        y_val = evalds['class'].map(mapping_dict)

        y_train = tf.keras.utils.to_categorical(y_train, num_classes=len(classes))
        y_val = tf.keras.utils.to_categorical(y_val, num_classes=len(classes))

        x_train = trainds.drop('class', axis=1)
        x_val = evalds.drop('class', axis=1)

        x_train = x_train.apply(self.normalize_column)
        x_val = x_val.apply(self.normalize_column)

        return (x_train, x_val), (y_train, y_val)

    def build_nn(self) -> tf.keras.Sequential:
        nn = tf.keras.Sequential([
            layers.Input(shape=(4,)),
            layers.Dense(128, activation='relu'),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dense(3, activation='softmax')
        ])

        lr_optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        nn.compile(optimizer=lr_optimizer,
                   loss='categorical_crossentropy',
                   metrics=['accuracy'])
        return nn

    def train_and_evaluate(self) -> History:
        model_export_path = os.path.join(self.output_dir, "savedmodel")
        checkpoint_path = os.path.join(self.output_dir, "checkpoints")
        tensorboard_path = os.path.join(self.output_dir, "tensorboard")

        if tf.io.gfile.exists(self.output_dir):
            tf.io.gfile.rmtree(self.output_dir)

        trainds = pd.read_csv(self.train_data_path)
        evalds = pd.read_csv(self.eval_data_path)

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
            batch_size=self.batch_size,
            verbose=2,
            callbacks=[checkpoint_cb, tensorboard_cb],
        )

        nn.save(model_export_path)
        return history
