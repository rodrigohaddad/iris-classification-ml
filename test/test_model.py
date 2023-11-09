import os
import unittest
import shutil

import numpy as np
import tensorflow as tf

from trainer.model import Model


class TestModelCase(unittest.TestCase):
    def setUp(self):
        self.current_directory = os.getcwd()
        # Remove 'test' from the join if running locally
        self.output = os.path.join(self.current_directory, 'test', 'output')
        self.eval = os.path.join(self.current_directory, 'test', 'input', 'eval.csv')
        self.train = os.path.join(self.current_directory, 'test', 'input', 'train.csv')
        self.saved_model = os.path.join(self.current_directory, 'test', 'output', 'savedmodel')

        os.mkdir(self.output)

    def tearDown(self):
        shutil.rmtree(self.output)

    def test_model(self):
        model = Model({'lr': 0.001,
                       'epochs': 15,
                       'batch_size': 20,
                       'output_dir': self.output,
                       'eval_data_path': self.eval,
                       'train_data_path': self.train,
                       },
                      )
        model.train_and_evaluate()

        trained_model = tf.saved_model.load(self.saved_model)

        r = trained_model(
            np.array([
                [6.3, 2.9, 5.6, 1.8],
                [6.5, 3.0, 5.8, 2.2],
                [7.6, 3.0, 6.6, 2.1],
                [4.9, 2.5, 4.5, 1.7]
            ], dtype=np.float32)
        )

        r = np.argmax(r, axis=-1)
        assert (r & [2, 2, 2, 2]).all()
