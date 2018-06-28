# Training SSD_Model

import tensorflow as tf
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib import slim

from datasets import dataset_factory
from config import train_config




def main():
    if not train_config.dataset_dir:
        raise ValueError('You must input the dataset directory')
    tf.logging.set_verbosity(tf.logging.DEBUG)
    with tf.Graph().as_default():
        # Config model deploy. Keep TF slim models structure.
        # Useful if want to need nultiple GPUs or servers in the future.
        # deploy_config =

        # with tf.device()

        # Create global step
        # global_step = slim.create_global_step()

        dateset = dataset_factory.get_dataset(train_config.dataset_name,
                                              train_config.dataset_split_name,
                                              train_config.dataset_dir)
        print(dateset.data_sources)


if __name__ == '__main__':
    main()