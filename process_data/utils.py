import os
import pandas as pd

from tqdm import tqdm

def read_raw_data(location: str) -> pd.DataFrame:
    """Read the OAS data specified in the directory location"""

    print('Reading data...')
    files = [location + f for f in os.listdir(location)]
    pds = []
    for f in tqdm(files):
        pds.append(pd.read_csv(f, header=1, dtype=str))
    df = pd.concat(pds)
    return df