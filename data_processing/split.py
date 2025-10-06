import os
import argparse
import pandas as pd
import json

from sklearn.model_selection import train_test_split

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
    parser.add_argument('train_location')
    parser.add_argument('val_location')
    parser.add_argument('test_location')
    return parser

# ==========

def compute_germline_data(df):
    from itertools import product
    
    germlines_dict = {'heavy_germline': [], 'light_germline': [], 'counter': []}

    for heavy_germline, data in df.groupby('heavy_germline'):
        for light_germline, data2 in data.groupby('light_germline'):
            germlines_dict['heavy_germline'].append(heavy_germline)
            germlines_dict['light_germline'].append(light_germline)
            germlines_dict['counter'].append(len(data2))

    germlines = pd.DataFrame(germlines_dict)

    all_germline_pairs = set(product(germlines['heavy_germline'].unique(), germlines['light_germline'].unique()))
    found_germline_pairs = set([(row['heavy_germline'], row['light_germline']) for _, row in germlines.iterrows()])
    not_found_germline_pairs = all_germline_pairs.difference(found_germline_pairs)

    zero_germlines = {'heavy_germline': [], 'light_germline': [], 'counter': []}

    for h, l in not_found_germline_pairs:
        zero_germlines['heavy_germline'].append(h)
        zero_germlines['light_germline'].append(l)
        zero_germlines['counter'].append(0)

    germlines = pd.concat([germlines, pd.DataFrame(zero_germlines)], axis=0).sort_values(by='counter', ascending=False)

    return germlines

def split_internal(germlines, sequences):
    # Select some heavy germlines exclusive to dataset 2
    heavy_germlines_dict = {'germline': [], 'counter': []}
    for g, data in germlines.groupby('heavy_germline'):
        heavy_germlines_dict['germline'].append(g)
        heavy_germlines_dict['counter'].append(data.sum()['counter'])
    heavy_germlines = pd.DataFrame(heavy_germlines_dict)

    heavy_germlines = heavy_germlines.sort_values(by='counter', ascending=False)

    lower = heavy_germlines['counter'].quantile(0.20)
    higher = heavy_germlines['counter'].quantile(0.80)

    selected_heavy_germlines = heavy_germlines[(heavy_germlines['counter'] > lower) & (heavy_germlines['counter'] < higher)]
    selected_heavy_germlines = selected_heavy_germlines.sample(int(len(selected_heavy_germlines)*0.1))

    # Select some light germlines exlcusive to dataset 2
    light_germlines_dict = {'germline': [], 'counter': []}
    for g, data in germlines.groupby('light_germline'):
        light_germlines_dict['germline'].append(g)
        light_germlines_dict['counter'].append(data.sum()['counter'])
    light_germlines = pd.DataFrame(light_germlines_dict)

    light_germlines = light_germlines.sort_values(by='counter', ascending=False)

    lower = light_germlines['counter'].quantile(0.20)
    higher = light_germlines['counter'].quantile(0.80)

    selected_light_germlines = light_germlines[(light_germlines['counter'] > lower) & (light_germlines['counter'] < higher)]
    selected_light_germlines = selected_light_germlines.sample(int(len(selected_light_germlines)*0.1))

    merged = sequences.merge(selected_heavy_germlines, left_on='heavy_germline', right_on='germline', how='left', indicator=True)
    merged = merged[merged['_merge'] == 'left_only'][sequences.columns]
    merged = merged.merge(selected_light_germlines, left_on='light_germline', right_on='germline', how='left', indicator=True)
    merged = merged[merged['_merge'] == 'left_only'][sequences.columns]
    to_split = merged

    df1, df2 = train_test_split(to_split, test_size=0.25, random_state=1234567890)

    df2 = pd.concat([
        df2, 
        pd.merge(sequences, selected_heavy_germlines['germline'], 
                 left_on='heavy_germline', right_on='germline')[sequences.columns],
        pd.merge(sequences, selected_light_germlines['germline'], 
                 left_on='light_germline', right_on='germline')[sequences.columns]
    ])

    df2 = df2.drop_duplicates()

    return df1, df2 

def split(location, train_location, val_location, test_location):
    print('Start splitting')
    representative = pd.read_csv(location, index_col=0)
    germlines_all = compute_germline_data(representative)
    trainval, test = split_internal(germlines_all, representative)
    test.to_csv(test_location)
    germlines_trainval = compute_germline_data(trainval)
    train, val = split_internal(germlines_trainval, trainval)
    train.to_csv(train_location)
    val.to_csv(val_location)
    print('Save training split:', train_location)
    print('Save validation split:', val_location)
    print('Save test split:', test_location)

# ========== MAIN

def main():
    parser = argparse.ArgumentParser(**get_argparse_argument())
    parser = generate_parser(parser)

    args = parser.parse_args()

    location = args.location
    train_location = args.train_location
    val_location = args.val_location
    test_location = args.test_location

    split(location, train_location, val_location, test_location)

if __name__ == '__main__':
    main()