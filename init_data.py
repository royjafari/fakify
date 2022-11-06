import os
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score

from fakify import curfake

def get_curfake_data(where):
    if not os.path.exists('data'):
        os.mkdir('data')

    url = f'data/{where}.pkl'
    
    
    if os.path.exists(url):
        return pd.read_pickle(url)
    else:
        ratios = np.linspace(0.001,50,10000)
        aucs = []
        for ratio in ratios:
            cf = curfake(auc=ratio, where=where)
            df = cf.get(10000)

            aucs.append(roc_auc_score(df.Label,df.Probability))
        
        aucs = np.array(aucs)

        df = pd.DataFrame([aucs,ratios]).T.rename(columns={0:'AUC',1:'Ratio'})
        df.to_pickle(url)
        return df

get_curfake_data('right')
get_curfake_data('left')