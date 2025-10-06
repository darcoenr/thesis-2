import argparse
import pandas as pd
import numpy as np

# ===== ARGPARSE GENERATION

def get_argparse_argument():
    parser_dict = {}
    return parser_dict

def generate_parser(parser):
    parser.add_argument('location_positive')
    parser.add_argument('location_legative')
    parser.add_argument('output_location')
    return parser

# ==========

# ===== FUNCTIONS

def merge_positive_and_negative(pos, neg):
    pos = pd.concat([pos[['pair_id', 'heavy', 'light']], 
                     pd.DataFrame({'class': np.zeros(len(pos), dtype=int)})],
                     axis=1)
    neg = pd.concat([neg[['pair_id', 'heavy', 'light']], 
                     pd.DataFrame({'class': np.ones(len(pos), dtype=int)})],
                     axis=1)
    data = pd.concat([pos, neg])
    return data.sample(len(data))

def merge(location_positive, location_negative, output_location):
    pos = pd.read_csv(location_positive, index_col=0)
    neg = pd.read_csv(location_negative)
    data = merge_positive_and_negative(pos, neg)
    data.to_csv(output_location)

# ==========

# ===== MAIN

def main():
    parser = argparse.ArgumentParser(**get_argparse_argument())
    generate_parser(parser)
    args = parser.parse_args()

    location_positive = args.location_positive
    location_negative = args.location_negative
    output_location = args.output_location

    merge(location_positive, location_negative, output_location)
    
if __name__ == '__main__':
    main()