import os
import sys
import openai
import pandas as pd
import numpy as np
import argparse
from sklearn.metrics import confusion_matrix,accuracy_score,ConfusionMatrixDisplay
from sklearn.metrics import mean_squared_error,mean_absolute_error,f1_score,log_loss,precision_score,recall_score,classification_report
from keras.losses import binary_crossentropy
from keras import backend
import glob

def read_predictions(directory):
    files = glob.glob(directory+"/*.csv")
    df_list = []
    for f in files:
        model=f.split("/")[-1].split(".")[0]
        df = pd.read_csv(f)
        df=df.rename(columns={'prompt':'scenario'})
        if model in ["delphi_gamma"]:
            df=df.assign(proba_1=df['class_probs.0']+df['class_probs.1'])
        df=df.loc[:,['context', 'condition', 'study', 'human.response','scenario', 'proba_1']]
        df=df.assign(model=model)
        model_type='_'.join(model.split('_')[:2])
        if model_type == 'shortv2_davinci':
            model_type='InstructGPT'
        elif model_type == 'long_davinci':
            model_type='GPT3'
        df=df.assign(model_type=model_type)
        df_list.append(df)   
        df_voting=pd.concat(df_list)
    return df_voting

def evaluate(df_voting):
    """Combine multiple prediction files. Average probability

    Parameters:
    dataframe with predictions
    Returns:
    Dataframe with result table

   """
    df_voting=df_voting.groupby(['context', 'condition', 'study', 'human.response','scenario','model_type']).proba_1.mean().reset_index()
    models=list()
    f1w_list=list()
    accuracy_list=list()
    precision_list=list()
    recall_list=list()
    conservativity_list=list()
    mae_list=list()
    ce_list=list()
    bluehouse_list=list()
    cannonball_list=list()
    lines_list=list()
    studies={'lines': list(), 'cannonball': list(), 'blue.house':list()}
    for m in df_voting.model_type.unique():
        df=df_voting.loc[df_voting.model_type==m,:]
        df=df.assign(human_response_binary=np.where(df['human.response']>0.5,1,0))
        df=df.assign(study=np.where(df.study.isin(['deli','deli.lines','snack.lines','snack']),'lines',df.study))
        df=df.assign(GPT3_response_probas_binary=np.where(df.proba_1>0.5,1,0))
        df=df.assign(proba_0=1-df.proba_1)
        mean_ce = backend.eval(binary_crossentropy(df['human.response'], df['proba_1']))
        models.append(m)
        gen=classification_report(df.loc[:,'human_response_binary'],
                                    df.loc[:,'GPT3_response_probas_binary'],digits=4,output_dict=True)
        cm=confusion_matrix(df.loc[:,'human_response_binary'],df.loc[:,'GPT3_response_probas_binary'])
        conserv=round(cm[1][0]/(cm[1][0]+cm[0][1]),4)

        for c in df['study'].unique():
            dd=classification_report(df.loc[df['study']==c,'human_response_binary'],
                                    df.loc[df['study']==c,'GPT3_response_probas_binary'],digits=4,output_dict=True)
            studies[c].append("{0:.2f}".format(100*dd['weighted avg']['f1-score']))

        ## davinci
        f1w="{0:.2f}".format(100*gen['weighted avg']['f1-score'])
        accuracy="{0:.2f}".format(100*gen['accuracy'])
        precision="{0:.2f}".format(100*gen['weighted avg']['precision'])
        recall="{0:.2f}".format(100*gen['weighted avg']['recall'])
        conservativity="{0:.2f}".format(100*conserv)
        mae=round(mean_absolute_error(df['human.response'], df.proba_1),3)
        ce=round(mean_ce,3)
        f1w_list.append(f1w)
        accuracy_list.append(accuracy)
        precision_list.append(precision)
        recall_list.append(recall)
        conservativity_list.append(conservativity)
        mae_list.append(mae)
        ce_list.append(ce)

    results=pd.DataFrame({'model':models,'F1':f1w_list,'Acc.':accuracy_list,
                         'Conserv':conservativity_list,
                         'MAE':mae_list,'CE':ce_list,'Line':studies['lines'],'Prop':studies['blue.house'],
                         'Cann.':studies['cannonball']})    

    return results

def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-d','--directory', nargs='?', help='directory with files with predictions',default='./results')
    parser.add_argument('-o','--output_file', nargs='?', help='output file name',default='./results_table.csv')
    args = parser.parse_args()
    
    df_voting=read_predictions(args.directory)
    df=evaluate(df_voting)
    df.to_csv(args.output_file,index=False)

if __name__ == '__main__':
    main()