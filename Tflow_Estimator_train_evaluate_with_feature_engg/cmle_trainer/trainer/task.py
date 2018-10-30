#Hasan - CMLE Model task.py 
import argparse
import json
import os

from . import model

import tensorflow as tf

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input_path_train',
        help = 'Input path pattern for training',
        required = True
    )
    parser.add_argument(
        '--input_path_eval',
        help = 'Input path pattern for eval',
        required = True
    )
    parser.add_argument(
        '--output_dir',
        help = 'GCS location to write checkpoints and export models',
        required = True
    )
    parser.add_argument(
        '--batch_size',
        help = 'Number of examples to compute gradient over.',
        type = int,
        default = 128
    )
    parser.add_argument(
        '--job-dir',
        help = 'this model ignores this field, but it is required by gcloud',
        default = 'junk'
    )

    ## TODO 1: add the new arguments here 
    parser.add_argument(
        '--train_examples',
        help = 'Number of examples (in thousands) to run the training job over. If this is more than actual # of examples available, it cycles through them. So specifying 1000 here when you have only 100k examples makes this 10 epochs.',
        type = int,
        default = 5000
    )    
    parser.add_argument(
        '--local_or_cmle',
        help = 'Specify a pattern that has to be in input files. For example 00001-of will process only one shard',
        default = 'local'
    )
    parser.add_argument(
        '--eval_steps',
        help = 'Positive number of steps for which to evaluate model. Default to None, which means to evaluate until input_fn raises an end-of-input exception',
        type = int,       
        default = None
    )
        
    ## parse all arguments
    args = parser.parse_args()
    arguments = args.__dict__

    # unused args provided by service
    arguments.pop('job_dir', None)
    arguments.pop('job-dir', None)

    ## assign the arguments to the model variables
    lom = arguments.pop('local_or_cmle')
    if lom == 'local':
        output_dir = os.getcwd() + arguments.pop('output_dir')
        model.INPUT_FILE     = os.getcwd() + arguments.pop('input_path_train')
        model.EVAL_FILE     = os.getcwd() +  arguments.pop('input_path_eval')
    else:
        output_dir = arguments.pop('output_dir')
        model.INPUT_FILE     = arguments.pop('input_path_train')
        model.EVAL_FILE     = arguments.pop('input_path_eval')
        
    model.BATCH_SIZE = arguments.pop('batch_size')
    model.TRAIN_STEPS = (arguments.pop('train_examples') * 1000) / model.BATCH_SIZE
    model.EVAL_STEPS = arguments.pop('eval_steps')    
    print ("Will train for {} steps using batch_size={}".format(model.TRAIN_STEPS, model.BATCH_SIZE))

    # Append trial_id to path if we are doing hptuning
    # This code can be removed if you are not using hyperparameter tuning
    print("Output path:  ", output_dir)
    print("Train path:  ", model.INPUT_FILE)
    print("Eval path:  ", model.EVAL_FILE)

    # Run the training job
    model.train_and_evaluate(output_dir)