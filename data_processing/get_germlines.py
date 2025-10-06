import os
import argparse
import pandas as pd
import json

from random import sample, seed
from tqdm import tqdm
from itertools import product
from collections import defaultdict

# ===== ARGPARSE GENERATION

def get_argparse_argument():
    parser_dict = {}
    return parser_dict

def generate_parser(parser):
    parser.add_argument('location')
    parser.add_argument('-o', default='../datasets', dest='output_location')
    return parser

# ==========

def germline_id(df, which='all'): 

    if which != 'all':
        df['heavy_germline'] = df['heavy_germline'].map(lambda x: x.split('-')[0])
        df['light_germline'] = df['light_germline'].map(lambda x: x.split('-')[0])
    
    print('Get germlines id...')
    heavy = df['heavy_germline'].unique()
    light = df['light_germline'].unique()

    print('Number of unique heavy combinations:', len(heavy))
    print('Number of unique light combinations:', len(light))
    print('Number of possibile heavy and light combinations:', len(heavy)*len(light))

    pairing_dict = {'heavy_germline': [], 'light_germline': [], 'pairing_index': [], 'counter': []}
    
    for i, ((h, l), data) in enumerate(df.groupby(['heavy_germline', 'light_germline'])):
        pairing_dict['heavy_germline'].append(h)
        pairing_dict['light_germline'].append(l)
        pairing_dict['pairing_index'].append(i)
        pairing_dict['counter'].append(len(data))

    seen_pairs = set(zip(pairing_dict['heavy_germline'], pairing_dict['light_germline']))
    all_pairs = set(product(heavy, light))
    never_seen_pairs = all_pairs.difference(seen_pairs)

    for i, (h, l) in enumerate(never_seen_pairs, start=pairing_dict['pairing_index'][-1] + 1):
        pairing_dict['heavy_germline'].append(h)
        pairing_dict['light_germline'].append(l)
        pairing_dict['pairing_index'].append(i)
        pairing_dict['counter'].append(0)

    df_pairing = pd.DataFrame(pairing_dict)
    df = df.merge(df_pairing, left_on=['heavy_germline', 'light_germline'], right_on=['heavy_germline', 'light_germline'])
    columns = list(df.columns)
    columns.remove('heavy_germline')
    columns.remove('light_germline')
    columns.remove('counter')
    df = df[columns]

    return df, df_pairing


def get_germlines(location, sequences_file, germline_file, which=all):
    df = pd.read_csv(location, index_col=0)
    df, pairs_df = germline_id(df, which)
    df.to_csv(sequences_file)
    pairs_df.to_csv(germline_file)
    print('Saved: {}, {}'.format(sequences_file, germline_file))

# ========== MAIN

def main():
    parser = argparse.ArgumentParser(**get_argparse_argument())
    parser = generate_parser(parser)

    args = parser.parse_args()

    location = args.location
    output_location = args.output_location

    get_germlines(location, output_location)

if __name__ == '__main__':
    main()