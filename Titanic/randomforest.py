import pandas as pd
import numpy as np
import csv as csv

from sklearn.datasets import make_blobs
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss
from sklearn import cross_validation
from sklearn.cross_validation import KFold
from sklearn.ensemble import (RandomTreesEmbedding, RandomForestClassifier,
                              GradientBoostingClassifier)
from sklearn.preprocessing import OneHotEncoder
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_curve
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

# Data cleanup
# TRAIN DATA
train_df = pd.read_csv('train.csv', header=0)        # Load the train file into a dataframe

# I need to convert all strings to integer classifiers.
# I need to fill in the missing values of the data and make it complete.

# female = 0, Male = 1
train_df['Gender'] = train_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Embarked from 'C', 'Q', 'S'
# Note this is not ideal: in translating categories to numbers, Port "2" is not 2 times greater than Port "1", etc.

# All missing Embarked -> just make them embark from most common place
if len(train_df.Embarked[ train_df.Embarked.isnull() ]) > 0:
    train_df.Embarked[ train_df.Embarked.isnull() ] = train_df.Embarked.dropna().mode().values

Ports = list(enumerate(np.unique(train_df['Embarked'])))    # determine all values of Embarked,
Ports_dict = { name : i for i, name in Ports }              # set up a dictionary in the form  Ports : index
train_df.Embarked = train_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)     # Convert all Embark strings to int

# All the ages with no data -> make the median of all Ages
median_age = train_df['Age'].dropna().median()
if len(train_df.Age[ train_df.Age.isnull() ]) > 0:
    train_df.loc[ (train_df.Age.isnull()), 'Age'] = median_age

# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
train_df = train_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 


# TEST DATA
test_df = pd.read_csv('test.csv', header=0)        # Load the test file into a dataframe

# I need to do the same with the test data now, so that the columns are the same as the training data
# I need to convert all strings to integer classifiers:
# female = 0, Male = 1
test_df['Gender'] = test_df['Sex'].map( {'female': 0, 'male': 1} ).astype(int)

# Embarked from 'C', 'Q', 'S'
# All missing Embarked -> just make them embark from most common place
if len(test_df.Embarked[ test_df.Embarked.isnull() ]) > 0:
    test_df.Embarked[ test_df.Embarked.isnull() ] = test_df.Embarked.dropna().mode().values
# Again convert all Embarked strings to int
test_df.Embarked = test_df.Embarked.map( lambda x: Ports_dict[x]).astype(int)


# All the ages with no data -> make the median of all Ages
median_age = test_df['Age'].dropna().median()
if len(test_df.Age[ test_df.Age.isnull() ]) > 0:
    test_df.loc[ (test_df.Age.isnull()), 'Age'] = median_age

# All the missing Fares -> assume median of their respective class
if len(test_df.Fare[ test_df.Fare.isnull() ]) > 0:
    median_fare = np.zeros(3)
    for f in range(0,3):                                              # loop 0 to 2
        median_fare[f] = test_df[ test_df.Pclass == f+1 ]['Fare'].dropna().median()
    for f in range(0,3):                                              # loop 0 to 2
        test_df.loc[ (test_df.Fare.isnull()) & (test_df.Pclass == f+1 ), 'Fare'] = median_fare[f]

# Collect the test data's PassengerIds before dropping it
ids = test_df['PassengerId'].values
# Remove the Name column, Cabin, Ticket, and Sex (since I copied and filled it to Gender)
test_df = test_df.drop(['Name', 'Sex', 'Ticket', 'Cabin', 'PassengerId'], axis=1) 


# The data is now ready to go. So lets fit to the train, then predict to the test!
# Convert back to a numpy array
train_data = train_df.values
test_data = test_df.values

print 'Training...'

'''
#fifthforest
#Encoder and Logestic Regression combined with Random Forest
rf = RandomForestClassifier(n_estimators = 100)
rf_enc = OneHotEncoder()
rf_lm = LogisticRegression()
rf = rf.fit( train_data[0::,1::], train_data[0::,0] )
rf_enc.fit(rf.apply(train_data[0::,1::]))
'''

X_train = train_data[0::,1::]
y_train = train_data[0::,0]
X_test = test_data
X_train, X_train_lr, y_train, y_train_lr = train_test_split(X_train,
                                                            y_train,
                                                            test_size=0.5)
# rf_lm.fit(rf_enc.transform(rf.apply(X_train_lr)), y_train_lr)

# y_pred_rf_lm = rf_lm.predict_proba(rf_enc.transform(rf.apply(X_test)))[:, 1]

#sixthforestt
#Encoder and Logestic Regression combined with Gradient Boosting Classifier
n_estimator = 10
grd = GradientBoostingClassifier(n_estimators=n_estimator)
grd_enc = OneHotEncoder()
grd_lm = LogisticRegression()
grd.fit(X_train, y_train)
grd_enc.fit(grd.apply(X_train)[:, :, 0])
grd_lm.fit(grd_enc.transform(grd.apply(X_train_lr)[:, :, 0]), y_train_lr)


# output = grd_lm.predict(test_data).astype(int)
#output = rf_lm.predict(rf_enc.transform(rf.apply(X_test))).astype(int)
output = grd_lm.predict(grd_enc.transform(grd.apply(X_test)[:, :, 0])).astype(int)


'''
#secondforest (in git)
#Cross Validation 
train_size = int(0.7*(train_data.shape[0]))
validation_size = train_data.shape[0] - train_size

X_train = train_data[0:train_size, 1::]
Y_train = train_data[0:train_size, 0]
X_validation = train_data[train_size::, 1::]
Y_validation = train_data[train_size::, 0]

clf = RandomForestClassifier(n_estimators =100)
clf.fit(X_train, Y_train)
sig_clf = CalibratedClassifierCV(clf, method = "sigmoid", cv="prefit")
sig_clf.fit(X_validation, Y_validation)

print 'Predicting...'
#output = sig_clf.predict(test_data).astype(int)
'''

'''
#thirdforest
#K-Fold Cross Validation
X = train_data[0::,1::]
y = train_data[0::,0]
k_fold = KFold(10, n_folds=3)
for k, (train,test) in enumerate(k_fold):
    forest.fit(X[train], y[train])
output = forest.predict(test_data).astype(int)
'''

'''
#fourthforest.csv
#Gradient Boosting Tree Used.
original_params = {'n_estimators': 1000, 'max_leaf_nodes': 4, 'max_depth': None, 'random_state': 2,
                   'min_samples_split': 5}
params = dict(original_params)
forest2 = GradientBoostingClassifier(**params)
forest2 = forest2.fit( train_data[0::,1::], train_data[0::,0] )
output = forest2.predict(test_data).astype(int)

'''
predictions_file = open("mysixthforest.csv", "wb")
open_file_object = csv.writer(predictions_file)
open_file_object.writerow(["PassengerId","Survived"])
open_file_object.writerows(zip(ids, output))
predictions_file.close()
print 'Done.'