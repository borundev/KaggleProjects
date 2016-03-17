import pandas as pd
import numpy as np

from sklearn.metrics import log_loss




def eval_wrapper(y, p):     
    return log_loss(y, p)


def cross_validate(clf,X,Y,n=3):
    import sklearn.cross_validation as cv
    
    print "#### Cross Validating "+str(n)+ " times"
    scores=[]
    
    for i in range(n):
        (X1, X2, Y1,Y2) = cv.train_test_split(X, Y, test_size=.25)
        clf=clf.fit(X1,Y1)
        prediction=clf.predict_proba(X2)[:,1]
        scores.append(eval_wrapper(Y2,prediction))
    
    return np.array(scores)


def write_submission_file(test_IDs,prediction,name="submission.csv"):

    final_result=pd.DataFrame([test_IDs,prediction],index=["ID","PredictedProb"])
    final_result=final_result.T
    final_result.ID=final_result.ID.astype(int)

    final_result.to_csv(name,index=False)

def Xgboost(X,Y,X_test,write=1):
    import xgboost as xgb
    gbm = xgb.XGBClassifier(max_depth=7, n_estimators=200, learning_rate=0.01) 
    print "fitting the model"
    gbm.fit(X, Y)
    print "train score is "+str(eval_wrapper(Y,gbm.predict_proba(X)[:,1]))
    print "cross validation score is "+str(cross_validate(gbm,X,Y))
    if(write):
        test_preds = gbm.predict_proba(X_test)[:,1]
        write_submission_file(test_IDs,test_preds,name="GBM.csv")

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')

Y=train.target.values
train.drop('target',axis=1,inplace=True)






train.drop('ID',axis=1,inplace=True)


test_IDs=test.ID.values
test.drop('ID',axis=1,inplace=True)

categorical=[]
for (train_name,train_series) in train.iteritems():
    categorical.append(train_name)

#j=train.columns[[type(k)==str for k in train.loc[1]]]

for i in categorical:
    train[i], tmp_indexer=pd.factorize(train[i])
    test[i]=tmp_indexer.get_indexer(test[i])

Xgboost(train.values,Y,test,1)





