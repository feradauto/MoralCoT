import os
import sys
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import os
import pandas as pd
import numpy as np
import argparse


def get_similarities(df_p1,sentence_embeddings):
    """Compute similarities for every pair of scenarios

    Parameters:
    df_p1: Dataframe with probabilities and texts
    sentence_embeddings: embeddings of the text
    Returns: dataframe with similarities

   """
    scores={'differences_pred':list(),'id_1':list(),'id_2':list()
           ,'human_response_1':list(),'human_response_2':list(),'differences_human':list(),'prompt_short_1':list(),
            'prompt_short_2':list(),'prediction_1':list(),'prediction_2':list(),
           'study_1':list(),'study_2':list(),'context_1':list(),'context_2':list(),
           'condition_1':list(),'condition_2':list(),'cosine_sim':list()}
    for i,d in df_p1.iterrows():
        for ii,dd in df_p1.iterrows():
            if (i>ii) and (d['human_response_binary']!=dd['human_response_binary']):
                scores['differences_pred'].append(abs(d['proba_1']-dd['proba_1']))
                cosine_sim=cosine_similarity([sentence_embeddings[i]],sentence_embeddings[ii:ii+1])[0][0]
                scores['id_1'].append(d['ID'])
                scores['id_2'].append(dd['ID'])
                scores['prompt_short_1'].append(d['prompt_short'])
                scores['prompt_short_2'].append(dd['prompt_short'])
                scores['human_response_1'].append(d['human.response'])
                scores['human_response_2'].append(dd['human.response'])
                scores['prediction_1'].append(d['proba_1'])
                scores['prediction_2'].append(dd['proba_1'])
                scores['study_1'].append(d['study'])
                scores['study_2'].append(dd['study'])
                scores['context_1'].append(d['context'])
                scores['context_2'].append(dd['context'])
                scores['condition_1'].append(d['condition'])
                scores['condition_2'].append(dd['condition'])
                scores['differences_human'].append(abs(d['human.response']-dd['human.response']))
                scores['cosine_sim'].append(cosine_sim)
    scores_df=pd.DataFrame(scores)
    scores_df=scores_df.loc[~scores_df.differences_pred.isna()].reset_index(drop=True)

    scores_df=scores_df.assign(score_similarity_negative=-scores_df.differences_pred)
    return scores_df

def get_scores_by_keyword(scores_df):
    """compute dogmatic score by keyword

    Parameters:
    scores_df : Dataframe with scores (similarity of text and score)
    Returns: dataframe with dogmatic score by keyword

   """
    scores_eq=scores_df.loc[scores_df.study_1==scores_df.study_2]
    cor_list=[]
    keywords=[]
    shape=[]
    uni=[]
    ## study
    for c in scores_eq.study_1.unique():
        df_cor=scores_eq.loc[ (scores_eq.study_1==c) ,['prompt_short_1','prompt_short_2','score_similarity_negative','cosine_sim']]
        cor=df_cor['score_similarity_negative'].corr(df_cor['cosine_sim'])
        s=df_cor.shape[0]
        s2=len(set(df_cor.prompt_short_2.unique()).union(set(df_cor.prompt_short_1.unique())))
        keywords.append(c)
        cor_list.append(cor)
        shape.append(s)
        uni.append(s2)
    ## condition
    scores_eq=scores_df.loc[scores_df.condition_1==scores_df.condition_2]
    for c in scores_eq.condition_1.unique():
        df_cor=scores_eq.loc[ (scores_eq.condition_1==c),['prompt_short_1','prompt_short_2','score_similarity_negative','cosine_sim']]
        cor=df_cor['score_similarity_negative'].corr(df_cor['cosine_sim'])
        s=df_cor.shape[0]
        s2=len(set(df_cor.prompt_short_2.unique()).union(set(df_cor.prompt_short_1.unique())))
        keywords.append(c)
        cor_list.append(cor)
        shape.append(s)
        uni.append(s2)

        ## context
    scores_eq=scores_df.loc[scores_df.context_1==scores_df.context_2]
    for c in scores_eq.context_1.unique():
        df_cor=scores_eq.loc[ (scores_eq.context_1==c),['prompt_short_1','prompt_short_2','score_similarity_negative','cosine_sim']]
        cor=df_cor['score_similarity_negative'].corr(df_cor['cosine_sim'])
        s=df_cor.shape[0]
        s2=len(set(df_cor.prompt_short_2.unique()).union(set(df_cor.prompt_short_1.unique())))
        keywords.append(c)
        cor_list.append(cor)
        shape.append(s)
        uni.append(s2)

    df_similarity=pd.DataFrame({'Keyword':keywords,'Dogmatic Score':cor_list, 'Samples': uni,'Combinations':shape})

    df_similarity=df_similarity.loc[(~df_similarity['Dogmatic Score'].isna())]

    df_similarity['Dogmatic Score']=df_similarity['Dogmatic Score'].apply(lambda x: round(x,3))

    corr=round(scores_df['score_similarity_negative'].corr(scores_df['cosine_sim']),3)
    samples=len(set(scores_df.prompt_short_2.unique()).union(set(scores_df.prompt_short_1.unique())))
    combinations=scores_df.shape[0]

    df_similarity=df_similarity.sort_values("Dogmatic Score",ascending=False)
    df_similarity.loc[-1] = ['Total', corr, samples,combinations]
    df_similarity['Dogmatic Score']=df_similarity['Dogmatic Score'].apply(lambda x: round(x,3))
    return df_similarity


def get_dogmatic_score(filename):
    """Gets dogmatic score

    Parameters:
    filename (str): Path to the file with scenarios
    Returns: dataframe with dogmatic score by keyword

   """
    model = SentenceTransformer('sentence-transformers/all-distilroberta-v1')
    df_p1=pd.read_csv(filename)
    df_p1=df_p1.assign(ID=df_p1.context+"_"+df_p1.condition+"_"+df_p1.study)
    df_p1=df_p1.assign(GPT3_response_probas_binary=np.where(df_p1.proba_1>0.5,1,0))
    df_p1=df_p1.assign(human_response_binary=np.where(df_p1['human.response']>0.5,1,0))

    sentence_embeddings = model.encode(df_p1.prompt_short)
    scores_df=get_similarities(df_p1,sentence_embeddings)
    df_similarity=get_scores_by_keyword(scores_df)
    
    return df_similarity

def main():
    argv = sys.argv[1:]
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument('-f','--file', nargs=1, help='input file name',required=True)
    parser.add_argument('-o','--output_file', nargs='?', help='output file name',default='./dogmatic_score.csv')
    args = parser.parse_args()
    
    df=get_dogmatic_score(args.file[0])
    df.to_csv(args.output_file,index=False)

if __name__ == '__main__':
    main()