import os
import sys
import openai
import pandas as pd
import numpy as np
import argparse
import glob
def get_prediction(folder):
    """Asks GPT3 if an action in a scenario is ok or not, using questions specifically designed for the scenario

    Parameters:
    filename (str): Path to the file with scenarios
    Returns:
    Dataframe with the following extra columns:
        full_prompt:    Input used for GPT3
        GPT3_response:  GPT3 response
        proba_1:        'yes' probability, the action is ok 
        proba_0:        'no' probability, the action is not ok 

   """
    files = glob.glob(folder+"/*.csv")
    df_list = []
    for f in files:
        df = pd.read_csv(f)
        df_list.append(df)
    df=pd.concat(df_list)
    
    return df

def main():
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-f','--folder', nargs=1, help='input folder name',required=True)
    parser.add_argument('-o','--output_file', nargs='?', help='output file name',default='./cot_features_join_0.csv')
    args = parser.parse_args()
    
    df=get_prediction(args.folder[0])
    df.to_csv(args.output_file,index=False)

if __name__ == '__main__':
    main()