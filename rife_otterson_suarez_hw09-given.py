import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from collections import Counter


# This function mutates, and also returns, the targetDF DataFrame.
# Mutations are based on values in the sourceDF DataFrame.
# You'll need to write more code in this function, to complete it.

def preprocess(targetDF, sourceDF):
    # For the Sex attribute, replace all male values with 0, and female values with 1.
    # (For this historical dataset of Titanic passengers, only "male" and "female" are listed for sex.)
    targetDF.loc[:, "Sex"] = targetDF.loc[:, "Sex"].map(lambda v: 0 if v=="male" else v)
    targetDF.loc[:, "Sex"] = targetDF.loc[:, "Sex"].map(lambda v: 1 if v=="female" else v)
    
    # Fill not-available age values with the median value.
    targetDF.loc[:, 'Age'] = targetDF.loc[:, 'Age'].fillna(sourceDF.loc[:, 'Age'].median())
    
	# -------------------------------------------------------------
	# Problem 4 code goes here, for fixing the error
    targetDF.loc[:, "Embarked"] = targetDF.loc[:, "Embarked"].map(
            lambda v: 0 if v=="C" else (1 if v=="Q" else (2 if v=="S" else v )))
    targetDF.loc[:, "Embarked"] = targetDF.apply(lambda row: targetDF.loc[:, "Embarked"].mode() 
    if np.isnan(row['Embarked']) else row['Embarked'], axis=1)

    # -------------------------------------------------------------
	# Problem 5 code goes here, for fixing the error
    targetDF.loc[:, 'Fare'] = targetDF.loc[:, 'Fare'].fillna(sourceDF.loc[:, 'Fare'].median())
   
	
# You'll need to write more code in this function, to complete it.
def buildAndTestModel():
    titanicTrain = pd.read_csv("data/train.csv")
    preprocess(titanicTrain, titanicTrain)
    titanicTest = pd.read_csv("data/test.csv")
    preprocess(titanicTest, titanicTrain)	    
    
    
    Xtrain = titanicTrain.loc[:, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    Ytrain = titanicTrain.loc[:, 'Survived']

    logReg = LogisticRegression(random_state=0, solver='liblinear').fit(Xtrain, Ytrain)
    print(model_selection.cross_val_score(logReg, Xtrain, Ytrain, cv=3).mean())

	#Problem 5 code goes here, to try the Kaggle testing set
    Xtest = titanicTest.loc[:, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    predictions = logReg.predict(Xtest)
    print(predictions, Counter(predictions), sep="\n")
    submitDF = pd.DataFrame(
                    {"PassengerId": titanicTest.loc[:,'PassengerId'],
                     "Survived": predictions
                     }
                )
    submitDF.to_csv("data/submitPredictions.csv", index=False)
    
def test09():
    buildAndTestModel()    
