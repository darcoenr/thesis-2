import os
import pathlib
import json
import argparse
import torch
import torch.nn as nn
import pandas as pd

import utils

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

def evaluate(name,
             val_data, batch_size,
             shuffle, subsample, frac, seed,
             model_name, model_checkpoint, tokenizer_checkpoint, eval_batches,
             save, out):
    
    location = r'../results/classification/{}/'.format(name)
    print('Location: {}'.format(location))

    # Get the data
    print('Retrieving the dataset...')
    val_ds, hyperpar = utils.get_dataset(val_data, subsample=subsample, frac=frac, seed=seed)
    val_dl = DataLoader(val_ds, batch_size=batch_size)

    # Load the model
    print('Load the model...')
    state_dict = torch.load('{}/model.pt'.format(location), weights_only=True)
    
    model = utils.load_model(model_name, checkpoint=model_checkpoint, state_dict=state_dict)

    # Get the tokenizer
    print('Retrieve the tokenizer...')
    if tokenizer_checkpoint is None: tokenizer_checkpoint = model_checkpoint
    tokenizer = utils.retrieve_tokenizer(tokenizer_checkpoint)

    criterion = nn.CrossEntropyLoss()

    results_eval, results_df = utils.evaluate(model, criterion, val_dl, 
                                              eval_batches=eval_batches,
                                              metrics={'accuracy': accuracy_score},
                                              tokenizer=tokenizer)
    if save:
        with open('{}/hyperpar_{}.json'.format(location, out), 'w') as f:
            json.dump(hyperpar, f)
        pd.DataFrame(results_eval).to_csv('{}/{}.csv'.format(location, out))
        results_df.to_csv('{}/{}_classification.csv'.format(location, out))

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

    evaluate(name, 
             val_data, subsample, batch_size,
             model_name, model_checkpoint, tokenizer_checkpoint, eval_batches,
             save, out)

"""
    location = r'../results/classification/{}/'.format(args.name)
    print('Location: {}'.format(location))

    # Get the data
    print('Retrieving the dataset...')
    val_ds, hyperpar = utils.get_dataset(args.val_data, subsample=args.subsample)
    val_dl = DataLoader(val_ds, batch_size=args.batch_size)

    # Load the model
    print('Load the model...')
    state_dict = torch.load(
        '{}/model.pt'.format(location), weights_only=True)
    model = utils.load_model(args.model_name, 
                             checkpoint=args.model_checkpoint, state_dict=state_dict)

    # Get the tokenizer
    print('Retrieve the tokenizer...')
    if args.tokenizer_checkpoint is None: args.tokenizer_checkpoint = args.model_checkpoint
    tokenizer = utils.retrieve_tokenizer(args.tokenizer_checkpoint)

    criterion = nn.CrossEntropyLoss()

    results_eval, results_df = utils.evaluate(model, criterion, val_dl, 
                                              eval_batches=args.eval_batches,
                                              metrics={'accuracy': accuracy_score},
                                              tokenizer=tokenizer)
    if args.save:
        with open('{}/hyperpar_{}.json'.format(location, args.out), 'w') as f:
            json.dump(hyperpar, f)
        pd.DataFrame(results_eval).to_csv('{}/{}.csv'.format(location, args.out))
        results_df.to_csv('{}/{}_classification.csv'.format(location, args.out))
"""

if __name__ == '__main__':
    main()