import numpy as np
import pandas as pd
import os

class linfake:
    def __init__(self,turning_point=(0.2,0.8),positive_ratio=0.5):

        assert turning_point[0]>0, 'turning point dimensions must be between 0 and 1, cannot be zero or one'
        assert turning_point[1]>0, 'turning point dimensions must be between 0 and 1, cannot be zero or one'

        assert turning_point[0]<1, 'turning point dimensions must be between 0 and 1, cannot be zero or one'
        assert turning_point[1]<1, 'turning point dimensions must be between 0 and 1, cannot be zero or one'

        assert positive_ratio >0, 'positive_ratio must be between 0 and 1, cannot be zero or one'
        assert positive_ratio <1, 'positive_ratio must be between 0 and 1, cannot be zero or one'

        self.turning_point = turning_point
        self.tupo_fpr = turning_point[0]
        self.tupo_tpr = turning_point[1]
        self.positive_ratio = positive_ratio

    def roc(self,x):
        FPR = self.tupo_fpr
        TPR = self.tupo_tpr
        
        return np.where(
            x<FPR,
            x*TPR/FPR,
            (TPR-1)/(FPR-1)*x +(FPR-TPR)/(FPR-1)
        )

    def get(self,n,random_state=None):
      
        df = pd.DataFrame(index = range(n), 
             columns = ['Label', 'Probability','Random']
            )

        df.Label = np.random.RandomState(random_state).binomial(1,self.positive_ratio,n)
        df.Random = np.random.RandomState(random_state).random(n)
        df.Probability = np.where(
                    df.Label == 0,
                    1-self.roc(df.Random),
                    self.roc(df.Random)
                                )
        return df.drop(columns = ['Random'])

class curfake:
    def __init__(self,auc=0.7,where='left',positive_ratio=0.5):

        #assert auc>=0.501, 'auc must be between 0.501 and 0.999'
        #assert auc<=0.999, 'auc must be between 0.001 and 0.999'

        assert positive_ratio >0, 'positive_ratio must be between 0 and 1, cannot be zero or one'
        assert positive_ratio <1, 'positive_ratio must be between 0 and 1, cannot be zero or one'

        self.auc = auc
        self.positive_ratio = positive_ratio
        self.where = where

        url = f'data/{where}.pkl'
        if not os.path.exists(url):
            self.ratio = auc
        else:
            self.ratio = self.get_ratio_per_auc(auc,where)


    def get_curfake_data(self,where):

        url = f'data/{where}.pkl'

        assert os.path.exists(url), 'init_data.py must be run first.'
        
        return pd.read_pickle(url)
    
    def get_ratio_per_auc(self,auc,where='left'):
        assert auc>=0.501, 'auc must be larger than or equal to 0.501'
        assert auc<=0.999, 'auc must be smaller than or equal to 0.999'

        df = self.get_curfake_data(where)

        BM = df.AUC == auc

        if sum(BM)==1:
            return df[BM].ratio.iloc[0]

        upper_auc_point = df[df.AUC<auc].AUC.max()
        lower_auc_point = df[df.AUC>auc].AUC.min()

        upper_ratio_point = df[df.AUC==upper_auc_point].Ratio.iloc[0]
        lower_ratio_point = df[df.AUC==lower_auc_point].Ratio.iloc[0]

        return (upper_ratio_point+lower_ratio_point)/2




    def roc_left(self,x):
        return x**self.ratio

    def roc_right(self,x):
        return (1- np.abs((x-1))/self.ratio)


    def get(self,n,random_state=None):

        df = pd.DataFrame(index = range(n), 
             columns = ['Label', 'Probability','Random']
            )

        df.Label = np.random.RandomState(random_state).binomial(1,self.positive_ratio,n)
        df.Random = np.random.RandomState(random_state).random(n)
        if(self.where =='left'):
            df.Probability = np.where(
                        df.Label == 0,
                        1-self.roc_left(df.Random),
                        self.roc_left(df.Random)
                                    )
        else:
            df.Probability = np.where(
                        df.Label == 0,
                        1-self.roc_right(df.Random),
                        self.roc_right(df.Random)
                                    )
        return df.drop(columns = ['Random'])



    