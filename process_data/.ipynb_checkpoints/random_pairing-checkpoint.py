import argparse
import numpy as np
import pandas as pd

from tqdm import tqdm
from itertools import product

# ===== ARGPARSE GENERATION

def get_argparse_argument():
    parser_dict = {}
    return parser_dict

def generate_parser(parser):
    parser.add_argument('location')
    parser.add_argument('-n', type=int)
    parser.add_argument('-o', default='../datasets/negative_samples_random.csv')
    parser.add_argument('--germline-ids-location', default='../datasets/germline_ids.csv')
    return parser

# ==========

# ===== FUNCTIONS

def sample_random_pairs(df, how_many):
    print('Sampling {} random pairs'.format(how_many))
    # Sample random index
    np.random.seed(0)
    index_heavy = np.random.randint(0, len(df), size=how_many)
    index_light = np.random.randint(0, len(df), size=how_many)
    
    # Boolean array indicating wether a couple of ids is in the original pairing.
    # This does no exclute the possibility of having pairs of sequences labeled as "not paired"
    # which are in fact paired in the original dataset. However it reduces the possibility to happen.
    
    not_equals = index_heavy != index_light
    index_list_heavy = [index_heavy[not_equals]]
    index_list_light = [index_light[not_equals]]
    while not all(not_equals):
        to_resample = sum(np.invert(not_equals))
        index_heavy = np.random.randint(0, len(df), size=to_resample)
        index_light = np.random.randint(0, len(df), size=to_resample)
        not_equals = index_heavy != index_light
        index_list_heavy.append(index_heavy[not_equals])
        index_list_light.append(index_light[not_equals])
    
    index_heavy = np.concatenate(index_list_heavy)
    index_light = np.concatenate(index_list_light)
    return index_heavy, index_light

def get_sequences(df, index_heavy, index_light):
    heavy_seqs = df[['heavy_germline', 'heavy']].iloc[index_heavy].reset_index(drop=True)
    #heavy_seqs = heavy_seqs.rename({'specie': 'specie_heavy'}, axis=1)
    
    light_seqs = df[['light_germline', 'light']].iloc[index_light].reset_index(drop=True)
    #light_seqs = light_seqs.rename({'specie': 'specie_light'}, axis=1)

    #eavy_specie = df['specie'].iloc[index_heavy]
    #light_specie = df['specie'].iloc[index_light]

    #specie = []
    #for hs, ls in zip(heavy_specie, light_specie):
    #    if hs == ls:
    #        specie.append(hs)
    #    else:
    #        specie.append('mixed')
    #specie = pd.DataFrame({'specie': specie})
    
    concat = pd.concat([heavy_seqs, light_seqs],#, specie],
                       axis=1, ignore_index=False)
    
    return concat                 

def random_pairing(location, output_location, n):
    df = pd.read_csv(location, index_col=0)

    print(df)
    
    # All the heavy and light sequences
    heavy_ids = df[['heavy_id', 'heavy']].drop_duplicates()
    light_ids = df[['light_id', 'light']].drop_duplicates()
    #germline_ids = pd.read_csv(germline_ids_location, index_col=0)
    #df = pd.merge(df, germline_ids)

    if n == None: n = len(df)
    index_heavy, index_light = sample_random_pairs(df, n)

    df = get_sequences(df, index_heavy, index_light)
    df = df.merge(germline_ids, 
                  left_on=['heavy_germline', 'light_germline'], 
                  right_on=['heavy_germline', 'light_germline'])

    # Get heavy id
    df = df.merge(heavy_ids, how='left')

    # Get light id
    df = df.merge(light_ids, how='left')
    
    # Get pairing id
    pair_seq = df[['heavy', 'light']].drop_duplicates()
    pairs_df = pd.DataFrame({'heavy': pair_seq['heavy'], 'light': pair_seq['light'], 'pair_id': list(range(len(pair_seq)))})
    df = df.merge(pairs_df)

    # Reorder the columns
    #df = df[['heavy_id', 'light_id', 'pair_id', 'pairing_index', 'heavy', 'light', 'specie']]
    df = df[['heavy_id', 'light_id', 'pair_id', 'heavy', 'light']]
    
    df.to_csv(output_location)
    print('Saved: {}'.format(output_location))

# ==========

# ===== MAIN

def main():
    parser = argparse.ArgumentParser(**get_argparse_argument())
    generate_parser(parser)
    args = parser.parse_args()

    location = args.location
    number = args.n
    output_location = args.o
    germline_ids_location = args.germline_ids_location

    random_pairing(location, output_location, germline_ids_location, number)
    
if __name__ == '__main__':
    main()