import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
from sklearn import cross_validation
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier,AdaBoostClassifier,RandomForestClassifier


train_original = pd.read_csv("train.csv")
target = train_original['TARGET']
train_original = train_original.drop(['ID','TARGET'],axis=1)

test_original = pd.read_csv("test.csv")
ids = test_original['ID'].values
test_original = test_original.drop(['ID'],axis=1)

train=train_original
test=test_original

categorical=[]
numeral=[]
for (train_name, train_series), (test_name, test_series) in zip(train.iteritems(),test.iteritems()):
    if train_name != test_name:
        print "train and test name are not the same"
        break
    name=train_name
    if train_series.dtype == 'O':
        categorical.append(name)
        train[name], tmp_indexer = pd.factorize(train[name])
        test[name] = tmp_indexer.get_indexer(test[name])
    else:
        numeral.append(name)
train_categorical=train[categorical]
train_numeral=train[numeral]
test_categorical=test[categorical]
test_numeral=test[numeral]


# impute missing data

from sklearn.preprocessing import Imputer
#imp = Imputer(missing_values=-1, strategy='median', axis=0)
#imp.fit(train_categorical)
#test_categorical=imp.transform(test_categorical)
#train_categorical=imp.transform(train_categorical)

imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
imp.fit(train_numeral)
test_numeral=imp.transform(test_numeral)
train_numeral=imp.transform(train_numeral)



X_train=np.hstack([train_numeral,train_categorical])
y_train=target

X_test=np.hstack([test_numeral,test_categorical])

from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)
fet_ind = np.argsort(rfc.feature_importances_)[::-1]
fet_imp = rfc.feature_importances_[fet_ind]

X_train=X_train[:,fet_ind[:50]]
X_test=X_test[:,fet_ind[:50]]





#print X_train.shape
#extc = ExtraTreesClassifier(n_estimators=700,max_features= 50,criterion= 'entropy',min_samples_split= 5,max_depth= 50, min_samples_leaf= 5,verbose=True,n_jobs=-1)
extc = ExtraTreesClassifier(n_estimators=750,max_features= 50,criterion= 'entropy',min_samples_split= 5,max_depth= 50, min_samples_leaf= 5, n_jobs = -1,verbose=True)      

print('Begin training')


extc.fit(X_train,y_train)

print('Predict...')
y_pred = extc.predict(X_test)

pd.DataFrame({"ID": ids, "TARGET": y_pred}).to_csv('extra_trees.csv',index=False)





