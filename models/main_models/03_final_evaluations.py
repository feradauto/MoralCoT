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

def random_model(filename,directory):    
    df=pd.read_csv(filename)
    df.loc[:,['context', 'condition', 'study', 'human.response', 'feature1',
       'feature2', 'feature3', 'feature1b', 'feature1c']]
    np.random.seed(seed=42)
    random_pred=np.random.uniform(0,1,df.shape[0])
    df['proba_1']=random_pred
    df.to_csv(directory+"/Random_Baseline.csv",index=False)
    return df

def majority_baseline(df_rand_m,directory):    
    df_rand_m=df_rand_m.assign(human_response_binary=np.where(df_rand_m['human.response']>0.5,1,0))
    random_majority=np.ones(df_rand_m.shape[0])*df_rand_m.human_response_binary.mean()
    df_rand_m['proba_1']=random_majority
    df_rand_m.to_csv(directory+"/Always_no.csv",index=False)
    return df_rand_m

def evaluate(directory):
    """Compute results table

    Parameters:
    directory with files with predictions
    Returns:
    Results table

   """
    files = glob.glob(directory+"/*.csv")
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
    df = []
    for f in files:
        model=f.split("/")[-1].split(".")[0]
        print(model)
        df = pd.read_csv(f)
        if 'delphi' in model:
            df=df.assign(GPT3_response_probas_binary=np.where(df.judgement_coded==-1,0,1))
            df=df.assign(proba_1=df['class_probs.0']+df['class_probs.1'])
        if model in ["GPT3","InstructGPT3","cot_general"]:
            df=df.rename(columns={'GPT3_response':'GPT3_final_response'})
            df=df.assign(GPT3_response_binary=
                    np.where(df['GPT3_final_response'].str.lower().str.contains("yes"),1,0))
            df=df.assign(proba_1=np.where(df.proba_1==1,0.99,df.proba_1))
            df=df.assign(proba_0=1-df.proba_1)
        df=df.assign(human_response_binary=np.where(df['human.response']>0.5,1,0))
        df=df.assign(study=np.where(df.study.isin(['deli','deli.lines','snack.lines','snack']),'lines',df.study))
        df=df.assign(GPT3_response_probas_binary=np.where(df.proba_1>0.5,1,0))
        if 'delphi' in model:
            df=df.assign(GPT3_response_probas_binary=np.where(df.judgement_coded==-1,0,1))
            df=df.assign(proba_1=df['class_probs.0']+df['class_probs.1'])
        mean_ce = backend.eval(binary_crossentropy(df['human.response'], df['proba_1']))

        models.append(model)
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
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-f','--file', nargs=1, help='input file name',required=True)
    parser.add_argument('-d','--directory', nargs='?', help='directory with files with predictions',default='./results')
    parser.add_argument('-o','--output_file', nargs='?', help='output file name',default='./results_table.csv')
    args = parser.parse_args()
    
    df_rand=random_model(args.file[0],args.directory)
    majority_baseline(df_rand,args.directory)
    
    df=evaluate(args.directory)
    df.to_csv(args.output_file,index=False)

if __name__ == '__main__':
    main()