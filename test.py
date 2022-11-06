from fakify import curfake, linfake
from sklearn.metrics import roc_auc_score


cf = curfake(auc=0.53,where='right', positive_ratio=0.1)
df = cf.get(100000)

print(roc_auc_score(df.Label,df.Probability))

lf = linfake(turning_point=(0.1,0.7),positive_ratio=0.1)
df = lf.get(100000)

print(roc_auc_score(df.Label,df.Probability))