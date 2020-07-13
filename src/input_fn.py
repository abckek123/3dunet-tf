"""
Data feeding function for train and test.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import tensorflow as tf
from src.data_utils import Dataset
import glob
import nibabel as nib
import numpy as np

IMAGE_HIGHT_MIN = 80
IMAGE_WIDTH_MIN = 80
IMAGE_HIGHT_MAX = 400
IMAGE_WIDTH_MAX = 400
IMAGE_HIGHT = IMAGE_HIGHT_MAX-IMAGE_HIGHT_MIN
IMAGE_WIDTH = IMAGE_WIDTH_MAX-IMAGE_WIDTH_MIN


def normalize(data):
    min = np.min(data)
    max = np.max(data)
    if not min == max:
        return (data - min)/(max - min)
    else:
        return data*0

def input_fn(training, params):
    """
    Simple input_fn for our 3D U-Net estimator, handling train and test data
    preparation.

    Args:
        training (bool): Whether we are training or testing.
        params (dict): Params for setting up the data. Expected keys are:
            max_scans (int): Maximum number of scans we see in any patient.
            train_img_size (int): Width and height of resized training images.
            batch_size (int): Number of of patient in each batch for training.
            num_classes (int): Number of mutually exclusive output classes.
            train_dataset_path (str): Path to pickled
                :class:`src.data_utils.Dataset` object.
            test_dataset_path (str): Path to pickled
                :class:`src.data_utils.Dataset` object.

    Returns:
        :class:`tf.dataset.Dataset`: An instantiated Dataset object.
    """
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # for training we use a batch number and pad each 3D scan to have equal
    # depth, width and height have already been set to 128 in preprocessing
    max_s = params['max_scans']
    w = h = params['train_img_size']
    if training:
        dataset = Dataset.load_dataset(
            os.path.join(package_root, params['train_dataset_path'])
        ).create_tf_dataset().shuffle(
            # we have 70 train examples, this will provide good shuffling
            buffer_size=70 
        ).repeat().padded_batch(
            batch_size=params['batch_size'],
            padded_shapes=(
                [max_s, w, h, 1], [max_s, w, h, params['num_classes']]
            )
        )

    # for testing we use the unscaled images with their original dims,
    # we still pad the depth dimension to max_s though
    else:
        # predicting a resized dataset, i.e. all have same width height?
        resized = 'resized' in params['test_dataset_path']
        dataset = Dataset.load_dataset(
            os.path.join(package_root, params['test_dataset_path'])
        ).create_tf_dataset(
            resized=resized
        ).padded_batch(
            # we have different sized test scans so we need batch 1
            batch_size=1,
            padded_shapes=(
                [max_s, None, None, 1],
                [max_s, None, None, params['num_classes']]
            )
        )

    iterator = tf.data.Iterator.from_structure(
        dataset.output_types,
        dataset.output_shapes
    )
    dataset_init_op = iterator.make_initializer(dataset)
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, dataset_init_op)
    next_element = iterator.get_next()

    # extremely hack way of getting tf.estimator to return labels at pred time
    # see https://github.com/tensorflow/tensorflow/issues/17824
    features = {'x': next_element[0], 'y': next_element[1]}
    return features, next_element[1]


def input_fn2(training, params):
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


    # for training we use a batch number and pad each 3D scan to have equal
    # depth, width and height have already been set to 128 in preprocessing
    max_s = params['max_scans']
    w = h = params['train_img_size']

    # max_s = 40
    # w = h = 512

    root_dir = '/home/zhangke/dataset'

    data_path = glob.glob(root_dir + "/**/ct.nii.gz", recursive=True)
    label_path = glob.glob(root_dir + "/**/label_ctv.nii.gz", recursive=True)
    if training:
        train_data_path = data_path[0:len(data_path)-6]
        train_label_path = label_path[0:len(label_path)-6]

        train_data_3d = []
        train_label_3d = []

        label_count = (0,0)
        for i in range(0,len(train_data_path)):

            data = nib.load(train_data_path[i]).get_data()
            label = nib.load(train_label_path[i]).get_data()

            depth = np.shape(data)[-1]

            depth_low  = depth//2-max_s//2
            width_low = np.shape(data)[0]//2-w//2

            train_data_3d.append(data[width_low:width_low+w,width_low:width_low+w,depth_low:depth_low+max_s])
            train_label_3d.append(label[width_low:width_low+w,width_low:width_low+w,depth_low:depth_low+max_s])

        train_data_3d = np.asarray(train_data_3d,dtype=np.float32)
        train_label_3d = np.asarray(train_label_3d , dtype= np.int32)

        train_data_3d = normalize(train_data_3d)

        train_data_3d = np.moveaxis(train_data_3d, -1, 1)
        train_label_3d = np.moveaxis(train_label_3d, -1, 1)

        train_data_3d = np.expand_dims(train_data_3d,-1)
        train_label_3d = np.expand_dims(train_label_3d,-1)



        dataset = tf.data.Dataset.from_tensor_slices((train_data_3d,train_label_3d)).shuffle(
            # we have 70 train examples, this will provide good shuffling
            buffer_size=24
        ).repeat().padded_batch(
            batch_size=params['batch_size'],
            padded_shapes=(
                [max_s, w, h, 1], [max_s, w, h, params['num_classes']]
            )
        )

    # for testing we use the unscaled images with their original dims,
    # we still pad the depth dimension to max_s though
    else:
        # predicting a resized dataset, i.e. all have same width height?
        test_data_path = data_path[len(data_path)-6:]
        test_label_path = label_path[len(label_path)-6:]

        test_data_3d = []
        test_label_3d = []
        for i in range(0,len(test_data_path)):

            data = nib.load(test_data_path[i]).get_data()
            label = nib.load(test_label_path[i]).get_data()

            width_low = np.shape(data)[0]//2-w//2
            depth = np.shape(data)[-1]

            if depth < max_s :
                zero = np.zeros([np.shape(data)[0],np.shape(data)[1],max_s-depth])
                data = np.dstack((zero,data ))
                label = np.dstack((zero,label ))
            

            depth = np.shape(data)[-1]
            depth_low  = depth//2-max_s//2
            test_data_3d.append(data[width_low:width_low+w,width_low:width_low+w,depth_low:depth_low+max_s])
            test_label_3d.append(label[width_low:width_low+w,width_low:width_low+w,depth_low:depth_low+max_s])




        test_data_3d = np.asarray(test_data_3d,dtype=np.float32)
        test_label_3d = np.asarray(test_label_3d , dtype= np.int32)
        test_data_3d = normalize(test_data_3d)

        test_data_3d = np.moveaxis(test_data_3d, -1, 1)
        test_label_3d = np.moveaxis(test_label_3d, -1, 1)

        test_data_3d = np.expand_dims(test_data_3d,-1)
        test_label_3d = np.expand_dims(test_label_3d,-1)


        dataset = tf.data.Dataset.from_tensor_slices((test_data_3d,test_label_3d)).padded_batch(
            # we have different sized test scans so we need batch 1
            batch_size=1,
            padded_shapes=(
                [max_s, None, None, 1],
                [max_s, None, None, params['num_classes']]
            )
        )

    iterator = tf.data.Iterator.from_structure(
        dataset.output_types,
        dataset.output_shapes
    )
    dataset_init_op = iterator.make_initializer(dataset)
    tf.add_to_collection(tf.GraphKeys.TABLE_INITIALIZERS, dataset_init_op)
    next_element = iterator.get_next()

    # extremely hack way of getting tf.estimator to return labels at pred time
    # see https://github.com/tensorflow/tensorflow/issues/17824
    features = {'x': next_element[0], 'y': next_element[1]}
    return features, next_element[1]
