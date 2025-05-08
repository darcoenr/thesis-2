import argparse
import pandas as pd
import numpy as np

from tqdm import tqdm
from itertools import product

import matplotlib.pyplot as plt

# ===== ARGPARSE GENERATION

def get_argparse_argument():
    parser_dict = {}
    return parser_dict

def generate_parser(parser):
    parser.add_argument('location')
    parser.add_argument('-n', type=int)
    parser.add_argument('-o', default='../datasets/negative_samples_germline.csv')
    parser.add_argument('--germline-ids-location', default='../datasets/germline_ids.csv')
    parser.add_argument('--alpha', type=int, default=0)
    return parser

# ==========

# ===== FUNCTIONS

def sample_zeros(df_pairing, how_many):
    # Get all the germline pairs having coocurrence equal to zero and select the relevant column.
    only_zeros = df_pairing[df_pairing['counter'] == 0][['heavy_germline', 'light_germline', 'pairing_index']]
    print('{} out of {} of only zeros'.format(len(only_zeros), len(df_pairing)))
    sampled_pairs = only_zeros.sample(how_many, replace=True)
    return sampled_pairs

# Sample zeros version two
def sample_zeros_v2(df, df_pairing, how_many, alpha=0):
    # Get all the germlines with zero occurrence
    only_zeros = df_pairing[df_pairing['counter'] == 0][['heavy_germline', 'light_germline', 'pairing_index']]
    print('{} out of {} of only zeros'.format(len(only_zeros), len(df_pairing)))
    # For each  pair of germile having zero cooccurrence
    # Compute how many different pairs of sequences can be picked.
    combinations_dict = {'pairing_index': [], 
                         'number_of_heavy': [], 
                         'number_of_light': [],
                         'number_of_combinations': []}
    total_sum = 0
    for _, row in tqdm(only_zeros.iterrows(), total=len(only_zeros)):
        number_of_heavy_germline = len(df[df['heavy_germline'] == row['heavy_germline']])
        number_of_light_germline = len(df[df['light_germline'] == row['light_germline']])
        number_of_combinations = number_of_heavy_germline * number_of_light_germline
        combinations_dict['pairing_index'].append(row['pairing_index'])
        combinations_dict['number_of_heavy'].append(number_of_heavy_germline)
        combinations_dict['number_of_light'].append(number_of_light_germline)
        # alpha: additive smoothing
        combinations_dict['number_of_combinations'].append(number_of_combinations + alpha)
        total_sum += number_of_combinations
    total_sum += alpha*len(only_zeros)
    combinations_dict['proportions'] = [x / total_sum for x in combinations_dict['number_of_combinations']]

    # Before continuing, a small comment.
    # The entries in combinations_dict['proportions'] can be interpreted as
    # the probability of picking a particular germline pair.
    
    #print(sorted(combinations_dict['proportions']))

###
#    probs = combinations_dict['proportions']
#    probs = sorted(probs, reverse=True)
#    print(probs)
#    plt.plot(probs)
#    plt.show()
###    
    rng = np.random.default_rng(0)
    sampled = rng.choice(combinations_dict['pairing_index'], how_many, replace=True, p=combinations_dict['proportions'])

    combinations_dict['how_many'] = []
    for pairing_index in combinations_dict['pairing_index']:
        combinations_dict['how_many'].append(sum(sampled == pairing_index))
    combinations = pd.DataFrame(combinations_dict)
    l = []
    for _, row in tqdm(combinations.iterrows(), total=len(combinations)):
        pairing_index = row['pairing_index']
        how_many = int(row['how_many'])
        l += [int(pairing_index) for _ in range(how_many)]
    return df_pairing.merge(pd.Series(l, name='pairing_index'))[['heavy_germline', 'light_germline', 'pairing_index']].sample(len(l))

def get_sequences(df, sampled_pairs):
    print(sampled_pairs)
    
    print('Retrieve sequences')
    sequences = {'heavy': [], 'light': [], 'pairing_index': []}
    for (h, l), data in tqdm(sampled_pairs.groupby(by=['heavy_germline', 'light_germline'])):
        h_df = df[df['heavy_germline'] == h]['heavy'].sample(len(data), replace=True, random_state=0)
        l_df = df[df['light_germline'] == l]['light'].sample(len(data), replace=True, random_state=0)

        #print(len(h_df))
        #print(len(l_df))
        #print(len(data))

        sequences['heavy'] += list(h_df)
        sequences['light'] += list(l_df)
        sequences['pairing_index'] += list(data['pairing_index'])
    generated_sequenced = pd.DataFrame(sequences)
    # This just shuffle the data.
    generated_sequenced = generated_sequenced.sample(len(generated_sequenced))
    return generated_sequenced

def germline_pairing(location: str, output_location: str, germline_ids_location: str, n:int, alpha):
    """Execute germline pairing"""
    
    df = pd.read_csv(location, index_col=0)
    heavy_ids = df[['heavy_id', 'heavy']].drop_duplicates()
    light_ids = df[['light_id', 'light']].drop_duplicates()
    germline_pairing_df = pd.read_csv(germline_ids_location,  index_col=0)
    df = pd.merge(df, germline_pairing_df)
    
    if n == None: n = len(df)
    #sampled_pairs = sample_zeros(germline_pairing_df, n)
    #print(sampled_pairs)
    sampled_pairs = sample_zeros_v2(df, germline_pairing_df, n, alpha)
    #print(sampled_pairs)

    df_original = df
    df = get_sequences(df, sampled_pairs).reset_index(drop=True)
    
    # Reorder the columns
    df = df[['pairing_index', 'heavy', 'light']]

    # Get heavy id
    df = df.merge(heavy_ids, how='left')

    # Get light id
    df = df.merge(light_ids, how='left')

    # Get specie
    heavy_specie = df['heavy_id'].to_frame().merge(df_original[['heavy_id', 'specie']]).drop_duplicates()
    heavy_specie = heavy_specie.rename({'specie': 'heavy_specie'}, axis=1)
    light_specie = df['light_id'].to_frame().merge(df_original[['light_id', 'specie']]).drop_duplicates()
    light_specie = light_specie.rename({'specie': 'light_specie'}, axis=1)
    df = df.merge(heavy_specie, how='left')
    df = df.merge(light_specie, how='left')
    
    specie = []
    for _, row in df.iterrows():
        if row['heavy_specie'] == row['light_specie']:
            specie.append(row['heavy_specie'])
        else:
            specie.append('mixed')
    df.insert(len(df.columns), 'specie', specie, True)

    # Get pairing id
    pair_seq = df[['heavy', 'light']].drop_duplicates()
    pairs_df = pd.DataFrame({'heavy': pair_seq['heavy'], 'light': pair_seq['light'], 'pair_id': list(range(len(pair_seq)))})
    df = df.merge(pairs_df)

    # Reorder the columns
    df = df[['heavy_id', 'light_id', 'pair_id', 'pairing_index', 'heavy', 'light', 'specie']]

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
    alpha = args.alpha

    germline_pairing(location, output_location, germline_ids_location, number, alpha)
    
if __name__ == '__main__':
    main()