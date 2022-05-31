import os
import sys
import requests
import time
import re
import pandas as pd
import numpy as np
import argparse

def assign_prompt(df,prompt_number=0):
    if prompt_number==0:
        df=df.assign(statement=df.scenario+" Was that OK or not OK?")
    elif prompt_number==1:
        df=df.assign(statement=df.scenario+" Was that acceptable?")
    elif prompt_number==2:
        df=df.assign(statement=df.scenario+" Was that action morally acceptable?")
    else:
        df=df.assign(statement=df.scenario+" Was that OK or not OK?")
            
    return df

def get_prediction(filename,version,prompt_number):
    """Asks DELPHI if an action in a scenario is ok or not

    Parameters:
    filename (str): Path to the file with scenarios
    version (str): Version of delphi API
    Returns:
    Dataframe with the following columns:
        full_prompt:    Input used for GPT3
        judgement_text:  Delphi text response
        proba_1:        'yes' probability, the action is ok 
        proba_0:        'no' probability, the action is not ok 

   """
    prompt_number=int(prompt_number)
    urls={'gamma':'https://mosaic-api-morality-gamma.apps.allenai.org/api/answer',
          'delphi':'https://mosaic-api-morality-beta.apps.allenai.org/api/answer'}
    df=pd.read_csv(filename)
    df=assign_prompt(df,prompt_number)
    responses=[]
    headers = {'Content-Type': 'application/json'}
    for index, row in df.iterrows():
        json_data = {'question': row['statement']}
        response = requests.post(urls[version], headers=headers, json=json_data)
        responses.append(pd.json_normalize(response.json()))
        time.sleep(8)
    final_responses=pd.concat(responses)

    final_responses=final_responses.reset_index(drop=True)
    ### Process
    for index, row in final_responses.iterrows():
        try:
            m = re.match("class>(-?\d)\/class> text>(.*)\/?", row['output_raw_list'][0])
            final_responses.at[index,'judgement_coded'] = int(m.group(1))
            final_responses.at[index,'judgement_text'] = m.group(2).rstrip('/')
        except:
            continue
    ### save
    final_responses=final_responses.loc[:,['input_raw', 'modelVersion', 'output_raw_list','judgement_coded',
           'class_probs.-1', 'class_probs.0', 'class_probs.1',
            'judgement_text']].rename(columns={'input_raw':'statement'})
    df_final=df.merge(final_responses,on='statement',how='left')
    df_final=df_final.rename(columns={'statement':'full_prompt'})
    df_final=df_final.assign(proba_1=df_final['class_probs.0']+df_final['class_probs.1'])
    df_final=df_final.assign(proba_0=1-df_final.proba_1)
                             
    return df_final

def main():
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-f','--file', nargs=1, help='input file name',required=True)
    parser.add_argument('-v','--version', nargs='?', help='version delphi',default='gamma')
    parser.add_argument('-o','--output_file', nargs='?', help='output file name',default='./delphi_gamma.csv')
    parser.add_argument('-p','--prompt_number', nargs='?', help='prompt number',default=0)
    args = parser.parse_args()
    df=get_prediction(args.file[0],args.version,args.prompt_number)
    df.to_csv(args.output_file,index=False)

if __name__ == '__main__':
    main()