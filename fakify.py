import numpy as np
import pandas as pd

class linfake:
    def __init__(self,turning_point=(0.2,0.8),positive_ratio=0.5):
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


    