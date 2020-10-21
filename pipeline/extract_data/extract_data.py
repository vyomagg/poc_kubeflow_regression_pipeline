# This script downloads the dataset locally, loads only a chunk of the dataset for the training.

import os
import pandas as pd
#import yaml
import argparse
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split


def fetch_datasets(train_samples=100, test_samples=20):

    df = pd.DataFrame(load_boston().data)
    df[len(df.columns)] = load_boston().target
    train_df, test_df = train_test_split(df, test_size= test_samples/(test_samples + train_samples))
    return train_df, test_df

def count_elements(df, df_type):
    ct = 0
    for x in df[0]:
        ct += 1
    print(f'{df_type} data containes {ct} data points.')

def save_datasets(train_df, test_df):
    train_df.to_pickle(output_train)
    test_df.to_pickle(output_test)
    count_elements(train_df, "Training")
    count_elements(test_df, "Testing")


if __name__ == '__main__':
    print("Starting the prepare stage...")

    #params = yaml.safe_load(open('params.yaml'))['extract']

    ## Initialize folders
    os.makedirs(os.path.join("app"), exist_ok=True)
    output_train = os.path.join('app', 'train.pkl')
    output_test = os.path.join('app', 'test.pkl')

    ## Fetch Parameters from pipeline
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_samples')
    parser.add_argument('--test_samples')
    args = parser.parse_args()

    ## Prepare Functions
    train_df, test_df = fetch_datasets(int(args.train_samples), int(args.test_samples))
    save_datasets(train_df, test_df)

    print('Prepare stage completed successfylly...')
