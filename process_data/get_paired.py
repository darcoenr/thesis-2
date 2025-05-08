import os
import argparse
import tqdm
import pandas as pd

# ===== ARGPARSE GENERATION

def get_argparse_argument():
    parser_dict = {}
    return parser_dict

def generate_parser(parser):
    parser.add_argument('location')
    parser.add_argument('-o', default='../datasets/paired_seq.csv')
    return parser

# ==========

# ===== FUNCTIONS

def get_paired(location, output_location):
    print('Extract paired sequences...')
    df = pd.read_csv(location, index_col=0)
    df = df[['heavy', 'light']].reset_index(drop=True)
    df.to_csv(output_location)
    print('Saved: {}'.format(output_location))

# ==========

# ===== MAIN

def main():
    parser = argparse.ArgumentParser(**get_argparse_argument())
    generate_parser(parser)
    args = parser.parse_args()

    location = args.location
    output_location = args.o

    get_paired(location, output_location)
    
if __name__ == '__main__':
    main()