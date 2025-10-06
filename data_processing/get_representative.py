import argparse
import pandas as pd

from tqdm import tqdm

# ===== ARGPARSE GENERATION

def get_argparse_argument():
    parser_dict = {}
    return parser_dict

def generate_parser(parser):
    parser.add_argument('location')
    parser.add_argument('paired_sequence_file')
    parser.add_argument('output_file')
    return parser

# ==========

def get_representative(location, paired_sequence_file, only_representative):
    print('Extracting only the representative sequences...')
    clusters = pd.read_csv(location, sep='\t', header=None).rename(
        { 0: 'representative', 1: 'sequences' },
        axis=1
    )
    sequences = pd.read_csv(paired_sequence_file, index_col=0)
    representative = clusters['representative'].drop_duplicates()
    representative = sequences.merge(representative, left_on='pair_id', right_on='representative', how='right')[sequences.columns]
    representative.to_csv(only_representative)
    print('Saved:', only_representative)

# ===== MAIN

def main():
    parser = argparse.ArgumentParser(**get_argparse_argument())
    generate_parser(parser)
    args = parser.parse_args()

    location = args.location
    paired_sequence_file = args.paired_sequence_file
    output_file = args.output_file

    get_representative(location, paired_sequence_file, output_file)
    
if __name__ == '__main__':
    main()