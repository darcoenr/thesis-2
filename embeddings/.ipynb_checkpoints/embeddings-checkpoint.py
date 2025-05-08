import argparse
import pathlib
import os
import json
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from transformers import BertModel
from torch.utils.data import DataLoader
from sklearn.manifold import TSNE

import sys
sys.path.insert(0, '../training/')
import utils
import models

# ===== ARGPARSE GENERATION

def get_argparse_argument():
    parser_dict = {}
    return parser_dict

def generate_parser(parser):
    # general
    parser.add_argument('--name', default='experiment')
    # data hyperpar
    parser.add_argument('--data', default='../datasets/train_germline.csv')
    parser.add_argument('--subsample', type=int)
    parser.add_argument('--frac', type=float)
    parser.add_argument('--batch-size', type=int, default=32)
    # model hyperpar
    parser.add_argument('--model-checkpoint', default='Exscientia/IgBert')
    parser.add_argument('--tokenizer-checkpoint')
    # figure parameters
    parser.add_argument('--figure-title', default='Germline')

    return parser

# ==========

def main():
    parser = argparse.ArgumentParser(**get_argparse_argument())
    generate_parser(parser)

    args = parser.parse_args()
    
    # Get the data
    print('Retrieving the datasets...')
    data, hyperpar = utils.get_dataset(args.data, subsample=args.subsample, frac=args.frac)
    data_dl = DataLoader(data, batch_size=args.batch_size)

    # Get BERT
    print('Retrieve transformer:')
    print('Model checkpoint: {}'.format(args.model_checkpoint))
    model = BertModel.from_pretrained(args.model_checkpoint, add_pooling_layer=False)

    # Get the tokenizer
    print('Retrieve the tokenizer...')
    if args.tokenizer_checkpoint is None: args.tokenizer_checkpoint = args.model_checkpoint
    tokenizer = utils.retrieve_tokenizer(args.tokenizer_checkpoint)

    # Compute embeddings
    print('Compute embeddings...')
    embeddings = utils.compute_avg_embeddings(model, tokenizer, data_dl)

    # Dimensionality reduction
    print('Reduce...')
    tsne_res = TSNE(2, perplexity=30, n_jobs=-1).fit_transform(embeddings)

    print('Plotting...')
    data_to_plot = pd.DataFrame({
        'x': tsne_res[:, 0], 
        'y': tsne_res[:, 1], 
        'class': [label for _, label in data]})

    fig, ax = plt.subplots()

    sns.scatterplot(data_to_plot, x='x', y='y', hue='class', palette='rocket_r', ax=ax, s=15)
    ax.set_title('Germline')
    ax.tick_params(axis='both', which='both', 
                   bottom=False, left=False, labelbottom=False, labelleft=False)
    ax.set(xlabel='', ylabel='')
    l = ax.legend()
    l.get_texts()[0].set_text('Positive')
    l.get_texts()[1].set_text('Negative')

    path = r'../results/embeddings/'.strip()
    if not pathlib.Path(path).exists():
        os.makedirs(path)
    with open('{}/hyperpar_{}.json'.format(path, args.name), 'w') as f:
        json.dump(hyperpar, f)
    
    fig.savefig('{}/{}.png'.format(path, args.name))

if __name__ == '__main__':
    main()