import os
import sys
import openai
import pandas as pd
import numpy as np
import argparse

def get_harms_benefits(filename,output_file):
    """Asks GPT3 for harms and benefits in an specific scenario

    Parameters:
    filename (str): Path to the file with scenarios
    output_file (str): Output path for the results

   """
    openai.api_key = os.getenv("OPENAI_API_KEY")
    preprompt="Consider this situation and answer the questions that follow:\nSituation: "
    f2="Who are positively or negatively affected by this action?"
    f3="Who will be worse off after this happens?  How much worse off will they be?"
    f4="Who will be better off after this happens?  How much better off will they be?"
    
    df=pd.read_csv(filename)
    df=df.assign(statement=preprompt+df.scenario+"\nQuestion: ")

    for i,d in df.iterrows():
        input_prompt=d['statement']+f2
        completion = openai.Completion.create(engine="text-davinci-002", prompt=input_prompt,temperature=0,max_tokens=100)
        df.loc[i,'first_response']=completion.choices[0].text
        input_2=input_prompt+completion.choices[0].text+"\nQuestion: "+f3+"\n"
        second_response = openai.Completion.create(engine="text-davinci-002", prompt=input_2,temperature=0,max_tokens=100)
        df.loc[i,'second_response']=second_response.choices[0].text
        input_3=input_2+second_response.choices[0].text+"\nQuestion: "+f4+"\n"
        third_response = openai.Completion.create(engine="text-davinci-002", prompt=input_3,temperature=0,max_tokens=100)
        df.loc[i,'third_response']=third_response.choices[0].text

    df_harms=df.loc[:,['context', 'condition', 'study','scenario','first_response','second_response', 'third_response']]

    df_harms=df_harms.rename(columns={'first_response':f2,
                             'second_response':f3,
                             'third_response':f4})
    
    df_harms[f2]=df_harms[f2].str.lstrip("\n")
    df_harms[f3]=df_harms[f3].str.lstrip("\n")
    df_harms[f4]=df_harms[f4].str.lstrip("\n")
    df_harms.to_csv(output_file,index=False)

def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-f','--file', nargs=1, help='input file name',required=True)
    parser.add_argument('-o','--output_file', nargs='?', help='output file name',default='./harms_benefits.csv')
    args = parser.parse_args()
    
    get_harms_benefits(args.file[0],args.output_file)

if __name__ == '__main__':
    main()