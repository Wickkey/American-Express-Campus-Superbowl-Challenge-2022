import pandas as pd
import numpy as np
from catboost import CatBoostClassifier, Pool, EShapCalcType, EFeaturesSelectionAlgorithm
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split, StratifiedKFold
import lightgbm as lgb
from sklearn.feature_selection import SelectFromModel,RFECV
import joblib
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

'''
Edit input_file_name and output_file_name here
'''

input_file_name = '6352880a54876_test_allx (1).csv' # Edit here
output_file_name = "New_test oofsquare_4classifierensemble2.csv" # Edit here

def denoise(df):
    dummies1 = pd.get_dummies(df.D_36, prefix = 'onehot_')
    dummies2 = pd.get_dummies(df.D_44, prefix = 'onehot_')

    df.drop(['D_36', 'D_44'], axis=1, inplace = True)

    
    df[dummies1.keys()] = dummies1.values
    df[dummies2.keys()] = dummies2.values

    for col in tqdm(df.columns):
        if col not in ['ID']:
            df[col] = np.floor(df[col]*100)
    return df

def addmetafeatureslevel1_Predict(models,X):
    coln = X.columns
    for item in models:
        model = models[item][0]

        meta_features = pd.DataFrame(model.predict_proba(X[coln])[:,:3])
        meta_features.columns = ['meta'+ item[:3] +'_0','meta'+ item[:3] +'_1','meta'+ item[:3] +'_2']

        X[meta_features.columns] = meta_features

    return X

def addmetafeatureslevel2_Predict(models,X):
    coln = X.columns
    for item in models:
        model = models[item][0]

        meta_features = pd.DataFrame(model.predict_proba(X[coln])[:,:3])
        meta_features.columns = ['meta2'+ item[:3] +'_0','meta2'+ item[:3] +'_1','meta2'+ item[:3] +'_2']

        X[meta_features.columns] = meta_features

    return X

new_data = pd.read_csv(input_file_name)
test_new = denoise(new_data)

id = test_new['ID']



clf_cat1 = joblib.load('Meta Models\CatB_1.pkl')
clf_lgb1 = joblib.load('Meta Models\LGB_1.pkl')
clf_cat2 = joblib.load('Meta Models\CatB_2.pkl')
clf_lgb2 = joblib.load('Meta Models\LGB_2.pkl')

catb_yhat,lgb_yhat, cat2_yhat, lgb2_yhat = [],[],[],[]


model1 = {'Cat' : [clf_cat1, catb_yhat], 'LGB':[clf_lgb1,lgb_yhat]}
model2 = {'Cat2_B':[clf_cat2,cat2_yhat],'LGB_2':[clf_lgb2,lgb2_yhat]}

test_new = addmetafeatureslevel1_Predict(model1,test_new)
test_new = addmetafeatureslevel2_Predict(model2,test_new)

clf_cat = joblib.load('Classifier Models/clf_cat.pkl')
clf_lgb1 = joblib.load('Classifier Models/clf_lgb1.pkl')
clf_lgb2 = joblib.load('Classifier Models/clf_lgb2.pkl')
clf_xgb1 = joblib.load('Classifier Models/clf_xgb.pkl')

estimators = [('clf_cat',clf_cat),('clf_lgb1',clf_lgb1),('clf_lgb2',clf_lgb2),('clf_xgb1',clf_xgb1)]

class Vote:
    def __init__(self, estimators, use_ = [True]*len(estimators)):
        self.estimators = estimators
        self.use_ = use_ 

    def predict_proba(self,X):
        ans = np.zeros((len(X),4))
        n = len(self.estimators)

        denom = 0
        for i in range(n):
            ans += (self.use_[i]*self.estimators[i][1].predict_proba(X))
            denom+=self.use_[i] 

        return ans/denom 

    def predict(self,X):
        self.ans = self.predict_proba(X)
        return np.argmax(self.ans,axis=1)

vote = Vote(estimators,use_ = [1,1,1,1])

submission = pd.DataFrame()
submission['ID'] = id
submission['Default_Flag'] = vote.predict(test_new)
submission.to_csv(output_file_name,header=None,index=None)


