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
from src.input_fn import input_fn2
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
    
    parser.add_argument(
        '-model_dir', 
        default='../models/base_model',
        required=True,
        help="Experiment directory containing params.json"
    )
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
    json_path = os.path.join(args.model_dir, 'params.json')
    assert os.path.isfile(
        json_path), "No json configuration file found at {}".format(json_path)
    params = Params(json_path).dict
    
    # check mode
    modes = ['train', 'train_eval', 'eval', 'predict']
    assert args.mode in modes, "mode has to be one of %s" % ','.join(modes) 
    
    # create logger, add loss and IOU to logging
    logger = set_logger(os.path.join(args.model_dir, 'train.log'))
    
    dataset = NiiDataset('/home/zhangke/dataset',params=params)
    # -------------------------------------------------------------------------
    # create model
    # -------------------------------------------------------------------------
    
    model = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=args.model_dir,
        params=params,
        config=tf.estimator.RunConfig(
            log_step_count_steps=params['display_steps']
        )
    )
    
    # -------------------------------------------------------------------------
    # train
    # -------------------------------------------------------------------------
    
    if args.mode in ['train_eval', 'train']:
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
    
    if args.mode == 'predict':
        dataset.load_test()
        predictions = model.predict(input_fn=lambda: dataset.input_fun(False))

        pred_file = os.path.join(args.model_dir,'predict')
        dataset.save_test_pred(prediction=predictions,savepath = pred_file)
        # extract predictions, only save predicted classes not probs
        logger.info('Predictions saved to: %s.' % pred_file)


if __name__ == '__main__':
    main(sys.argv[1:])
