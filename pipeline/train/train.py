import pandas as pd
import argparse
import pickle
import os
# import yaml
from sklearn.linear_model import LinearRegression

def load_data(pkl_filepath):
    df = pd.read_pickle(pkl_filepath)
    return df

def save_model(model):
    with open(output_model, 'wb') as file:
        pickle.dump(model, file)

def linear_regr_model(train_df,fit_intercept,normalize,n_jobs,copy_X):
    # Create Linear Regression Object
    lm2 = LinearRegression(fit_intercept=fit_intercept, normalize=normalize, copy_X=copy_X, n_jobs=n_jobs)
    Y2 = train_df.pop(len(train_df.columns) - 1)
    X2 = train_df

    # Fit (Train) the model
    lm2.fit(X2, Y2)

    print("Intercept for the model is", lm2.intercept_, "and the scope is", lm2.coef_)

    # Save model
    save_model(lm2)

if __name__ == '__main__':
    print("Starting the training stage...")

    # params = yaml.safe_load(open('params.yaml'))['train']

    ## Initialize folders
    os.makedirs(os.path.join("model", "Regression_checkpoints"), exist_ok=True)
    output_model = os.path.join("model", "Regression_checkpoints", "best.pkl")

    ## Fetch Parameters from pipeline
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_train')
    parser.add_argument('--fit_intercept')
    parser.add_argument('--normalize')
    parser.add_argument('--n_jobs')
    parser.add_argument('--copy_X')
    args = parser.parse_args()

    ## Train Functions
    train_df = load_data(args.input_train)
    linear_regr_model(train_df, bool(args.fit_intercept) , bool(args.normalize), int(args.n_jobs), bool(args.copy_X))

    print("Training stage completed...")