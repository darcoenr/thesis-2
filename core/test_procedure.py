import os
import pathlib
import json
import argparse
import torch
import torch.nn as nn
import pandas as pd

import core.utils as utils

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score

# ===== ARGPARSE GENERATION

def get_argparse_argument():
    parser_dict = {}
    return parser_dict

def generate_parser(parser):
    # general
    parser.add_argument('--verbose', default=0, type=int)
    parser.add_argument('--name', default='experiment')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--out', default='evaluation')
    # data hyperpar
    parser.add_argument('--val-data', default='../datasets/val_germline.csv')
    parser.add_argument('--shuffle', default=False)
    parser.add_argument('--subsample', type=int)
    parser.add_argument('--frac', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--batch-size', type=int, default=32)
    # model hyperpar
    parser.add_argument('--model-name', default='average')
    parser.add_argument('--model-checkpoint', default='Exscientia/IgBert')
    parser.add_argument('--tokenizer-checkpoint')
    parser.add_argument('--state-dict-location')
    # epochs hyperpar  
    parser.add_argument('--eval-batches', type=int, default=None)

    return parser

def test_procedure(model_location,
             data_location,
             results_location,
             batch_size, shuffle, subsample, frac, seed,
             model_name, model_checkpoint, tokenizer_checkpoint, eval_batches,
             save, out):
        
    # Create results directory
    utils.create_dir_if_not_exists(results_location)
    print('Results location: {}'.format(results_location))

    # Get the data
    print('Retrieving the dataset...')
    val_ds, hyperpar = utils.get_test_dataset(data_location, 
                                         shuffle=shuffle, subsample=subsample, frac=frac, seed=seed)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    # Load the model
    print('Load the model...')
    state_dict = torch.load('{}'.format(model_location), weights_only=True,
                            map_location='cuda' if torch.cuda.is_available() else 'cpu')
    model = utils.load_model(model_name, checkpoint=model_checkpoint, state_dict=state_dict)

    # Get the tokenizer
    print('Retrieve the tokenizer...')
    if tokenizer_checkpoint is None: tokenizer_checkpoint = model_checkpoint
    tokenizer = utils.retrieve_tokenizer(tokenizer_checkpoint)

    results_df = utils.evaluate_test(model, val_dl, 
                               eval_batches=eval_batches,
                               tokenizer=tokenizer)
    if save:
        with open('{}/hyperpar_{}.json'.format(results_location, out), 'w') as f:
            json.dump(hyperpar, f)
        #pd.DataFrame(results_eval).to_csv('{}/{}.csv'.format(location, out))
        results_df.to_csv('{}/{}_prediction.csv'.format(results_location, out))

# ==========

def main():
    parser = argparse.ArgumentParser(**get_argparse_argument())
    generate_parser(parser)
    
    args = parser.parse_args()

    name = args.name
    save = args.save
    out = args.out

    val_data = args.val_data
    subsample = args.subsample
    batch_size = args.batch_size
    eval_batches = args.eval_batches

    model_name = args.model_name
    model_checkpoint = args.model_checkpoint
    tokenizer_checkpoint = args.tokenizer_checkpoint

    test_procedure(name, 
             val_data, subsample, batch_size,
             model_name, model_checkpoint, tokenizer_checkpoint, eval_batches,
             save, out)

if __name__ == '__main__':
    main()