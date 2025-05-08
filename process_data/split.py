import argparse
import os
import pathlib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ===== ARGPARSE GENERATION

def get_argparse_argument():
    parser_dict = {}
    return parser_dict

def generate_parser(parser):
    parser.add_argument('location_pos')
    parser.add_argument('location_neg')
    parser.add_argument('pos_pairing_loc')
    parser.add_argument('neg_pairing_loc')
    #parser.add_argument('--germline-ids-location', default='../datasets/germline_ids.csv')
    parser.add_argument('--proportions', nargs=3, type=int, default=[50, 25, 25])
    parser.add_argument('--output_path', default='../datasets/')
    parser.add_argument('--additional-name')
    return parser

# ==========

# ========== FUNCTIONS

def get_sequences(df, pos_pairing_loc, neg_pairing_loc, pos_cluster, neg_cluster, only_representative):

    # For positive
    # Get all positive cluster representative
    df_pos = df[df['class'] == 0]

    if not only_representative:
        # From the cluster representative, get all the elements in the cluster
        df_pos = pd.merge(df_pos, pos_cluster, left_on='seq', right_on='representative')[['elements', 'class']]
        df_pos = df_pos.rename({'elements': 'pair_id'}, axis=1)
    else:
        df_pos = df_pos.rename({'seq': 'pair_id'}, axis=1)
    
    
    # Get heavy and light
    pos_pairing = pd.read_csv(pos_pairing_loc, index_col=0).drop_duplicates()
    df_pos = pd.merge(df_pos, pos_pairing, how='left')[['pair_id', 'heavy', 'light', 'class']]
        
    # Repeat for negative
    df_neg = df[df['class'] == 1]
    
    if not only_representative:
        # From the cluster representative, get all the elements in the cluster
        df_neg = pd.merge(df_neg, neg_cluster, left_on='seq', right_on='representative')[['elements', 'class']]
        df_neg = df_neg.rename({'elements': 'pair_id'}, axis=1)
    else:
        df_neg = df_neg.rename({'seq': 'pair_id'}, axis=1)
    
    neg_pairing = pd.read_csv(neg_pairing_loc, index_col=0).drop_duplicates()
    df_neg = pd.merge(df_neg, neg_pairing, how='left')[['pair_id', 'heavy', 'light', 'class']]

    return pd.concat([df_pos, df_neg]) 
    
def split(location_pos, location_neg, 
          pos_pairing_loc, neg_pairing_loc,
          only_representative,
          proportions, output_location, additional_name):

    # Read the clustering tsv files.
    # Those files have the format:
    # 0 | 1
    # x | x
    # x | y
    # x | z
    # Where x is the representative of the cluster
    # and x, y and z are the elements in the cluster.
    pos_clusters = pd.read_csv(location_pos, sep='\t', header=None)
    pos_clusters = pos_clusters.rename({0: 'representative', 1: 'elements'}, axis=1)
    neg_clusters = pd.read_csv(location_neg, sep='\t', header=None)
    neg_clusters = neg_clusters.rename({0: 'representative', 1: 'elements'}, axis=1)

    # Get all the clustering representative.
    pos = pos_clusters['representative'].unique()
    neg = neg_clusters['representative'].unique()

    print('There are {} paired sequences with {} clusters.'.format(len(pos_clusters), len(pos)))
    print('There are {} negative paired sequences with {} clusters.'.format(len(neg_clusters), len(neg)))

    df = pd.concat([
        pd.DataFrame({'seq': pos, 'class': np.zeros(len(pos), dtype=int)}),
        pd.DataFrame({'seq': neg, 'class': np.ones(len(neg), dtype=int)})
    ], axis=0)

    train, valtest = train_test_split(df, train_size=proportions[0]/sum(proportions), 
                                      stratify=df['class'], random_state=0)
    val, test = train_test_split(valtest, train_size=proportions[1]/sum(proportions[1:]),
                                 stratify=valtest['class'], random_state=0)


    train_df = get_sequences(train, pos_pairing_loc, neg_pairing_loc, pos_clusters, neg_clusters, only_representative)
    val_df = get_sequences(val, pos_pairing_loc, neg_pairing_loc, pos_clusters, neg_clusters, only_representative)
    test_df = get_sequences(test, pos_pairing_loc, neg_pairing_loc, pos_clusters, neg_clusters, only_representative)

    # This just shuffle the data
    train_df = train_df.sample(len(train_df))
    val_df = val_df.sample(len(val_df))
    test_df = test_df.sample(len(test_df))

    if not pathlib.Path(output_location).exists():
        os.makedirs(output_location)

    print('{}/train{}.csv'.format(output_location, '' if additional_name == '' else '_' + additional_name))

    train_df.to_csv('{}/train{}.csv'.format(output_location, '' if additional_name == '' else '_' + additional_name))
    val_df.to_csv('{}/val{}.csv'.format(output_location, '' if additional_name == '' else '_' + additional_name))
    test_df.to_csv('{}/test{}.csv'.format(output_location, '' if additional_name == '' else '_' + additional_name))

# ==========

# ==========

def main():
    parser = argparse.ArgumentParser(**get_argparse_argument())
    parser = generate_parser(parser)

    args = parser.parse_args()

    location_pos = args.location_pos
    location_neg = args.location_neg
    pos_pairing_loc = args.pos_pairing_loc
    neg_pairing_loc = args.neg_pairing_loc

    split(location_pos, location_neg, 
          pos_pairing_loc, neg_pairing_loc,
          args.proportions, args.output_path, args.additional_name)

if __name__ == '__main__':
    main()


"""

    df = pd.read_csv(location, index_col=0)
    train, valtest = train_test_split(df, train_size=train_size, random_state=0, stratify=df['label'])
    val, test = train_test_split(valtest, train_size=val_size, random_state=0, stratify=valtest['label'])

    if output_location[-1] != '/': output_location += '/'
    if additional_name is not None: 
        additional_name = '_' + additional_name
    else:
        additional_name = ''

    train_location = output_location + 'train' + additional_name + '.csv'
    val_location = output_location + 'val' + additional_name + '.csv'
    test_location = output_location + 'test' + additional_name + '.csv'

    train.to_csv(train_location)
    val.to_csv(val_location)
    test.to_csv(test_location)

"""