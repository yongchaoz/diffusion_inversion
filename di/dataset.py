import os
import sys
from absl import logging

import tqdm
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds


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


def load_data(ds, img_shape, resolution=32):
    if resolution <= 64:
        batch_size = 5000
    else:
        batch_size = 1000
        
    size = len(ds)
    logging.info('Dataset size: {}'.format(size))
    if None in img_shape:
        x = np.zeros(shape=(size, resolution, resolution, 3), dtype=np.uint8)
    else:
        x = np.zeros(
            shape=(size, img_shape[0], img_shape[1], img_shape[2]), dtype=np.uint8)

    if None in img_shape:
        ds = ds.map(lambda x, y: (center_crop(
            x, resolution), y), tf.data.AUTOTUNE)
    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    y_list = []
    count = 0
    for x_batch, y_batch in tqdm.tqdm(tfds.as_numpy(ds), desc='Process the data'):
        num = x_batch.shape[0]
        x_processed = np.array(x_batch)
        x[count:count + num] = x_processed
        y_list.append(y_batch)
        count += num

    return x, np.concatenate(y_list, axis=0)


def configure_dataloader(ds, batch_size, x_transform=None, y_transform=None, train=False, shuffle=False, seed=0, resolution=None):
    if y_transform is None:
        def y_transform(x): return x
    else:
        y_transform = y_transform

    ds = ds.cache()
    if train:
        ds = ds.repeat()
    if shuffle:
        ds = ds.shuffle(16 * batch_size, seed=seed)

    if resolution is not None:
        ds = ds.map(lambda x, y: (tf.clip_by_value(
            tf.image.resize(x, [resolution, resolution], 'bilinear'),
            0, 255), y), tf.data.AUTOTUNE)

    ds = ds.map(lambda x, y: (tf.cast(x, tf.float32), y), tf.data.AUTOTUNE)

    if x_transform:
        ds = ds.map(lambda x, y: (x_transform(
            x), y_transform(y)), tf.data.AUTOTUNE)
    else:
        ds = ds.map(lambda x, y: x, y_transform(y), tf.data.AUTOTUNE)

    ds = ds.batch(batch_size=batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def get_dataset(config, return_raw=False, resolution=None, train_only=True):
    dataset_name = config.name
    data_dir = config.data_dir

    if dataset_name in ['imagenette']:
        split = ['train', 'validation']
    else:
        split = ['train', 'test']

    if resolution is None:
        if dataset_name in ['cifar10', 'cifar100']:
            resolution = 32
        elif dataset_name in ['stl10']:
            resolution = 96
        elif dataset_name in ['imagenette']:
            resolution = 256

    ds_builder = tfds.builder(dataset_name, data_dir=data_dir)
    
    ds_builder.download_and_prepare()

    img_shape = ds_builder.info.features['image'].shape
    num_train, num_test = ds_builder.info.splits[split[0]
                                                 ].num_examples, ds_builder.info.splits[split[1]].num_examples
    num_classes, class_names = ds_builder.info.features[
        'label'].num_classes, ds_builder.info.features['label'].names

    ds_train, ds_test = ds_builder.as_dataset(split=split, as_supervised=True)

    print('Number of training samples: {}'.format(num_train))
    print('Number of test samples: {}'.format(num_test))
    sys.stdout.flush()

    with config.unlocked():
        config.img_shape = (resolution, resolution,
                            3) if None in img_shape else img_shape
        config.num_classes = num_classes
        config.class_names = class_names
        config.train_size = num_train
        config.test_size = num_test

    x_train, y_train = load_data(ds_train, img_shape, resolution)

    if train_only:
        return x_train, y_train

    x_test, y_test = load_data(ds_test, img_shape, resolution)

    if return_raw:
        return x_train, y_train, x_test, y_test
    else:
        ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train))
        ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
        return ds_train, ds_test