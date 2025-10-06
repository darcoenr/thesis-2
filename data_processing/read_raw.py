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
    parser.add_argument('-o', default='../datasets/sequences.csv', dest='output_location')
    parser.add_argument('--subsample', type=int)
    parser.add_argument('--only-human', action='store_true')
    parser.add_argument('--which-germline')
    parser.add_argument('--store-specie', action='store_false')
    return parser

# ==========

def read_raw_data(location, subsample, only_human):
    """Read the OAS data specified in the directory location"""

    # Get all the files in the direcory location
    print('Reading data...')
    if location[-1] != '/': location += '/'
    files = [location + f for f in os.listdir(location)]
    print('Found {} files.'.format(len(files)))

    # Split filenames by species
    by_species_files = defaultdict(list)
    for f  in files:
        # Read metadata of the file
        meta = json.loads(','.join(pd.read_csv(f, nrows=0).columns))
        by_species_files[meta['Species']].append(f)
    for key, item in by_species_files.items():
        print('{:15}{:10}'.format(key + ':', len(item)))

    # Get only the human ones
    if only_human:
        files = by_species_files['human']

    # Subsample
    if subsample is not None:
        print('Sample {} files'.format(subsample))
        seed(0)
        files = sample(files, subsample)

    # Read the data
    pds = []
    for f in tqdm(files):
        specie = json.loads(','.join(pd.read_csv(f, nrows=0).columns))['Species']
        df = pd.read_csv(f, header=1, dtype=str)
        df.insert(len(df.columns), 'specie', [specie]*len(df), True)
        pds.append(df)
    df = pd.concat(pds)
    return df

def filter_data(df, which_germline):
    """Filter the dataframe"""
    
    print('Filtering the data')
    n_rows_initial = len(df)
    print('Initial number of rows: {}'.format(n_rows_initial))
    
    # Select the most important columns
    aa_seq = ['sequence_alignment_aa_heavy', 'sequence_alignment_aa_light']
    if which_germline == 'all':
        germline = ['v_call_heavy', 'd_call_heavy', 'j_call_heavy', 'v_call_light', 'j_call_light']
    elif which_germline is not None:
        germline = ['{}_call_heavy'.format(which_germline), '{}_call_light'.format(which_germline)]
    else:
        germline = []
    
    anarci = [c for c in df.columns if 'ANARCI' in c]
    informative_columns = aa_seq + germline + anarci + ['specie']
    df = df[informative_columns]

    # Filter out all the rows whose ANARCI status contains 'Shorter'.
    anarci_query = "(~ANARCI_status_heavy.str.contains('Shorter'))"
    anarci_query += " & (~ANARCI_status_light.str.contains('Shorter'))"
    df = df.query(anarci_query)

    # Keep only the germline, sequences and specie
    df = df[aa_seq + germline + ['specie']]

    # Drop the duplicates and nan rows.
    df = df.drop_duplicates()
    df = df.dropna()
    print('Removed {} rows (-{:.3f}%), new number of rows: {}.'.format(n_rows_initial - len(df), 
                                                                       (1.0 - len(df)/n_rows_initial)*100,
                                                                       len(df)))
    return df

def assign_sequence_id(df):
    """Assign an unique seuqnce to each heavy chain, light chain and pair of heavya nd light"""
    
    print('Assining ids...')
    
    heavy_seq = df['heavy'].unique()
    print('Number of unique heavy: {}'.format(len(heavy_seq)))
    heavy_df = pd.DataFrame({'heavy': heavy_seq, 'heavy_id': list(range(len(heavy_seq)))})
    df = df.merge(heavy_df)

    light_seq = df['light'].unique()
    print('Number of unique light: {}'.format(len(light_seq)))
    light_df = pd.DataFrame({'light': light_seq, 'light_id': list(range(len(light_seq)))})
    df = df.merge(light_df)

    pair_seq = df[['heavy', 'light']].drop_duplicates()
    print('Number of unique pairs:  {}'.format(len(pair_seq)))
    pairs_df = pd.DataFrame({'heavy': pair_seq['heavy'], 'light': pair_seq['light'], 'pair_id': list(range(len(pair_seq)))})
    df = df.merge(pairs_df)

    return df

def clean_germlines(df):
    
    print('Cleaning the germlines...')
    germline_columns = [c for c in df.columns if 'call' in c]#['heavy_germline', 'light_germline']
    def remove_allele(str):
        return str.split('*')[0]
    def remove_ig(str):
        return str[2:]
    def remove_gene(str):
        return str.split('-')[0]
    def remove_s(str):
        return str.split('S')[0]
    def remove_d(str):
        return str.split('D')[0]
    for c in [c for c in germline_columns if c != 'd_call_heavy']:
        df[c] = df[c].map(remove_allele).map(remove_ig).map(remove_gene).map(remove_s).map(remove_d)
    if 'd_call_heavy' in germline_columns:
        df['d_call_heavy'] = df['d_call_heavy'].map(remove_allele).map(remove_ig).map(remove_gene).map(remove_s)

    def concat_germlines(x):
        concatd = ''
        for xx in x.iloc[:-1]:
            concatd += (xx + '-')
        concatd += x.iloc[-1]
        return concatd
    
    heavy_germlines = [g for g in germline_columns if 'heavy' in g]
    light_germlines = [g for g in germline_columns if 'light' in g]
    
    heavy_concatenation = df[heavy_germlines].apply(concat_germlines, axis=1)
    light_concatenation = df[light_germlines].apply(concat_germlines, axis=1)

    #print(heavy_concatenation)
    #print(light_concatenation)

    df = pd.concat([df, heavy_concatenation, light_concatenation], axis=1).rename(
        {
            0: 'heavy_germline',
            1: 'light_germline'
        },
        axis=1
    )
    
    return df
    
def read_raw(location, output_location, *, subsample=None, only_human=False, which_germline=None, store_specie=True):
    df = read_raw_data(location, subsample, only_human)
    df = filter_data(df, which_germline)
    df = df.rename({
        'sequence_alignment_aa_heavy': 'heavy',
        'sequence_alignment_aa_light': 'light'
    }, axis=1)
    df = df.reset_index(drop=True)
    df = assign_sequence_id(df)
    if which_germline is not None:
        df = clean_germlines(df)
    
    columns = ['heavy_id', 'light_id', 'pair_id']
    columns += ['heavy_germline', 'light_germline'] if which_germline is not None else []
    columns += ['heavy', 'light']
    columns += ['specie'] if store_specie else []

    
    df = df[columns]
    df = df.drop_duplicates()

    df.to_csv(output_location)
    print('Saved: {}'.format(output_location))

# ========== MAIN

def main():
    parser = argparse.ArgumentParser(**get_argparse_argument())
    parser = generate_parser(parser)

    args = parser.parse_args()

    location = args.location
    output_location = args.output_location
    subsample = args.subsample
    only_human = args.only_human
    which_germline = args.which_germline
    store_specie = args.store_specie

    read_raw(location, output_location, 
             subsample=subsample, only_human=only_human, 
             which_germline=which_germline,
             store_specie=store_specie)

if __name__ == '__main__':
    main()