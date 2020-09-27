"""
Train 3D U-Net network, for prostate MRI scans.

Ideas taken from:
https://github.com/cs230-stanford/cs230-code-examples/tree/master/tensorflow/vision

and

https://github.com/tensorflow/models/blob/master/samples/core/
get_started/custom_estimator.py
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import pickle
import argparse

import tensorflow as tf

from src.model_fn import model_fn
from src.utils import Params, set_logger

from niiDataset import NiiDataset
import numpy as np
import nibabel as nb
import re

def arg_parser(args):
    """
    Define cmd line help for main.
    """
    
    parser_desc = "Train, eval, predict 3D U-Net model."
    parser = argparse.ArgumentParser(description=parser_desc)
    
    # parser.add_argument(
    #     '-model_dir', 
    #     default='../models/base_model',
    #     required=True,
    #     help="Experiment directory containing params.json"
    # )

    parser.add_argument(
        '-mode', 
        default='train_eval',
        help="One of train, train_eval, eval, predict."
    )

    parser.add_argument(
        '-pred_ix',
        nargs='+',
        type=int,
        default=[1],
        help="Space separated list of indices of patients to predict."
    )
    
    # parse input params from cmd line
    try:
        return parser.parse_args(args)
    except:
        parser.print_help()
        sys.exit(0)


def main(argv):
    """
    Main driver/runner of 3D U-Net model.
    """
    
    # -------------------------------------------------------------------------
    # setup
    # -------------------------------------------------------------------------

    # set the random seed for the whole graph for reproductible experiments
    tf.set_random_seed(42)

    # load the parameters from model's json file as a dict
    args = arg_parser(argv)
    json_path = os.path.join('models/base_model', 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path).dict
    
    # check mode
    train_batch = params['train_batch']
    modelPath = params['model_dir']
    if not os.path.exists(modelPath):
        os.makedirs(modelPath)

    modes = ['train', 'train_eval', 'eval', 'predict','train_pred']
    assert args.mode in modes, "mode has to be one of %s" % ','.join(modes) 
    
    # create logger, add loss and IOU to logging
    logger = set_logger(os.path.join(modelPath, 'train.log'))
    
    os.environ['CUDA_VISIBLE_DEVICES'] = params['CUDA_VISIBLE_DEVICES']
    # dataset = NiiDataset('/home/zhangke/datasets',params=params)
    dataset = NiiDataset(params=params)
    # -------------------------------------------------------------------------
    # create model
    # -------------------------------------------------------------------------
    model = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=modelPath,
        params=params,
        config=tf.estimator.RunConfig(
            log_step_count_steps=params['display_steps'],
            session_config= tf.ConfigProto(
            #     allow_soft_placement=True,
            #     log_device_placement=True,
            
            )
        )
    )
    
    # -------------------------------------------------------------------------
    # train
    # -------------------------------------------------------------------------
    
    if args.mode in ['train_pred','train_eval', 'train']:
        dataset.load_train()
        model.train(
            input_fn=lambda: dataset.input_fun(True),
            max_steps=params['max_train_steps']
        )
    
    # -------------------------------------------------------------------------
    # evaluate
    # -------------------------------------------------------------------------
    
    if args.mode in ['train_eval', 'eval']:
        dataset.load_test()
        model.evaluate(input_fn=lambda: dataset.input_fun(False))
    
    # -------------------------------------------------------------------------
    # predict
    # -------------------------------------------------------------------------
    
    if args.mode in ['train_pred','predict']:
        dataset.load_test()
        predictions = model.predict(input_fn=lambda: dataset.input_fun(False))

        predictPath = params['predict_dir']
        if not os.path.exists(predictPath):
            os.makedirs(predictPath)
        dataset.save_test_pred(prediction=predictions,savepath = predictPath)
        # extract predictions, only save predicted classes not probs
        logger.info('Predictions saved to: %s.' % predictPath)


if __name__ == '__main__':
    main(sys.argv[1:])
