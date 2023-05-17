import os
import sys
import tqdm
import fire
import functools
import ml_collections
from absl import logging

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from dataset import get_dataset  # NOQA

tf.config.experimental.set_visible_devices([], "GPU")


def D(**kwargs):
    return ml_collections.ConfigDict(initial_dictionary=kwargs)


def center_crop(x, resolution):
    shape = tf.shape(x)
    h, w = shape[0], shape[1]
    size = tf.minimum(h, w)
    begin = tf.cast([h - size, w - size], tf.float32) / 2.0
    begin = tf.cast(begin, tf.int32)
    begin = tf.concat([begin, [0]], axis=0)  # Add channel dimension.
    x = tf.slice(x, begin, [size, size, 3])
    x = tf.image.resize_with_pad(
        x, resolution, resolution, method='area', antialias=True)
    return x


def compute_channel_mean_std_ds(ds, img_shape, resolution=32, batch_size=1000):
    if None in img_shape:
        dim = resolution * resolution
    else:
        dim = functools.reduce(lambda x, y: x * y, img_shape[:-1], 1)

    # ds = ds.map(lambda x, y: tf.cast(
    #     x, dtype='float32') / 255.0, tf.data.AUTOTUNE)
    ds = ds.map(lambda x, y: tf.cast(
        x, dtype='float32'), tf.data.AUTOTUNE)
    if None in img_shape:
        ds = ds.map(lambda x: center_crop(x, resolution), tf.data.AUTOTUNE)
    ds = ds.map(lambda x: tf.reshape(
        x, shape=(dim, img_shape[-1])), tf.data.AUTOTUNE)
    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    mean = np.zeros(shape=(img_shape[-1],))
    var = np.zeros(shape=(img_shape[-1],))
    count = 0

    for x_batch in tqdm.tqdm(tfds.as_numpy(ds), desc='Compute mean with batch size: {}'.format(batch_size)):
        mean = mean + np.sum(x_batch, axis=(0, 1))
        count += x_batch.shape[0]

    mean = 1.0 / (count * dim) * mean

    for x_batch in tqdm.tqdm(tfds.as_numpy(ds), desc='Compute variance with batch size: {}'.format(batch_size)):
        var = var + np.sum(np.square(x_batch - mean), axis=(0, 1))

    std = np.sqrt(1.0 / (count * dim) * var)

    logging.info(
        'Total number of data: {}, mean: {}, std: {}'.format(count, mean, std))

    return mean, std


def main(dataset_name, data_dir, resolution, batch_size):
    dataset_config = D(
        name=dataset_name,
        data_dir=data_dir)

    x_train, y_train = get_dataset(
        dataset_config, return_raw=True, resolution=resolution, train_only=True)

    ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    mean, std = compute_channel_mean_std_ds(
        ds, (resolution, resolution, 3), resolution=resolution, batch_size=batch_size)
    print('Mean: {}, Std: {}'.format(mean, std))
    
    
if __name__ == "__main__":
    fire.Fire(main)
    # python compute_statistics.py --dataset_name=imagenette --data_dir=$HOME/tensorflow_datasets --resolution=256 --batch_size=1000