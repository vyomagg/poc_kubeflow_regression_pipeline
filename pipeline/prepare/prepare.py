import os
import argparse
from scipy import stats
import pandas as pd
#import yaml

def read_dataset(input_path):
    df = pd.read_pickle(input_path)
    return df

def feature_selection(train_df,test_df, co_relation_threshold):
    temp_train_df = pd.DataFrame()
    temp_test_df = pd.DataFrame()
    ct = 0

    for x in train_df:
        pearson_coef, p_value = stats.pearsonr(train_df[x], train_df[len(train_df.columns) - 1])
        print(f'Pearson coefficient for {x} columns is {pearson_coef}')

        #Feature selection on the bases of pearson_co_relation.
        if abs(pearson_coef) < co_relation_threshold or (len(train_df.columns) - 1 == x):
            temp_train_df[ct] = train_df[x]
            temp_test_df[ct] = test_df[x]
            ct += 1

    print(f'{ct} out of {len(train_df.columns)} features selected.')
    save_as_pkl(temp_train_df, temp_test_df)

def save_as_pkl(train_df,test_df):
    train_df.to_pickle(output_train)
    test_df.to_pickle(output_test)


if __name__ == '__main__':
    print("Starting the prepare stage...")

    #params = yaml.safe_load(open('params.yaml'))['prepare']

    ## Initialize folders
    os.makedirs(os.path.join("app"), exist_ok=True)
    output_train = os.path.join('app', 'train_prep.pkl')
    output_test = os.path.join('app', 'test_prep.pkl')

    ## Fetch Parameters from pipeline
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_train')
    parser.add_argument('--input_test')
    parser.add_argument('--co_relation_threshold')
    args = parser.parse_args()

    ## Prepare Functions
    train_df = read_dataset(args.input_train)
    test_df = read_dataset(args.input_test)
    feature_selection(train_df, test_df, args.co_relation_threshold)

    print('Prepare stage completed successfylly...')