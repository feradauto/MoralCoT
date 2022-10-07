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

def random_model(filename,directory,seed=42):    
    df=pd.read_csv(filename)
    df.loc[:,['context', 'condition', 'study', 'human.response', 'feature1',
       'feature2', 'feature3', 'feature1b', 'feature1c']]
    np.random.seed(seed=seed)
    random_pred=np.random.uniform(0,1,df.shape[0])
    df['proba_1']=random_pred
    df.to_csv(directory+"/Random_Baseline_"+str(seed)+".csv",index=False)
    return df

def read_predictions(directory):
    files = glob.glob(directory+"/*.csv")
    df_list = []
    for f in files:
        model=f.split("/")[-1].split(".")[0]
        df = pd.read_csv(f)
        df=df.rename(columns={'prompt':'scenario'})
        if model in ["delphi_gamma"]:
            df=df.assign(proba_1=df['class_probs.0']+df['class_probs.1'])
        df=df.loc[:,['study','context','condition', 'human.response','scenario', 'proba_1']]
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
    """Combine multiple prediction files. 
    Parameters:
    dataframe with predictions
    Returns:
    Dataframe with result table

   """
    df_voting=df_voting.groupby(['study', 'human.response','scenario','model','model_type']).proba_1.mean().reset_index()
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
    models_t=list()
    studies={'lines': list(), 'cannonball': list(), 'blue.house':list()}
    for m in df_voting.model.unique():
        df=df_voting.loc[df_voting.model==m,:]
        df=df.assign(human_response_binary=np.where(df['human.response']>0.5,1,0))
        df=df.assign(study=np.where(df.study.isin(['deli','deli.lines','snack.lines','snack']),'lines',df.study))
        df=df.assign(GPT3_response_probas_binary=np.where(df.proba_1>0.5,1,0))
        df=df.assign(proba_0=1-df.proba_1)
        mean_ce = backend.eval(binary_crossentropy(df['human.response'], df['proba_1']))
        models.append(m)
        m_type=df_voting.loc[df_voting.model==m,:].model_type.unique()[0]
        models_t.append(m_type)
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

    results=pd.DataFrame({'model':models,'model_type':models_t,'F1':f1w_list,'Acc.':accuracy_list,
                         'Conserv':conservativity_list,
                         'MAE':mae_list,'CE':ce_list,'Line':studies['lines'],'Prop':studies['blue.house'],
                         'Cann.':studies['cannonball']})    

    return results

def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-d','--directory', nargs='?', help='directory with files with predictions',default='./results')
    parser.add_argument('-o','--output_results', nargs='?', help='output folder for the agg results',default='./')
    parser.add_argument('-f','--file', nargs=1, help='input file name',default="../../input_data/complete_file.csv")
    args = parser.parse_args()
    filename=args.file[0]
    df1=random_model(filename,args.directory,seed=42)
    df2=random_model(filename,args.directory,seed=0)
    df3=random_model(filename,args.directory,seed=1)
    df4=random_model(filename,args.directory,seed=2)
    
    df_voting=read_predictions(args.directory)
    df_voting.to_csv(args.output_results+"/all_predictions.csv",index=False)
    
    df_voting=df_voting.loc[:,['study', 'human.response', 'scenario','proba_1', 'model', 'model_type']]
    
    df=evaluate(df_voting)
    ## mean prediction
    df=df.assign(F1=df.F1.apply(pd.to_numeric))
    df=df.assign(Conserv=df.Conserv.apply(pd.to_numeric))
    df=df.assign(Line=df.Line.apply(pd.to_numeric))
    df=df.assign(Prop=df.Prop.apply(pd.to_numeric))
    df['Acc.']=df['Acc.'].apply(pd.to_numeric)
    df['Cann.']=df['Cann.'].apply(pd.to_numeric)
    mean=np.round(df.groupby(['model_type']).mean(),2)
    std=np.round(df.groupby(['model_type']).std(),2)
    
    
    mean.to_csv(args.output_results+"/predictions_mean.csv")
    std.to_csv(args.output_results+"/predictions_std.csv")

if __name__ == '__main__':
    main()