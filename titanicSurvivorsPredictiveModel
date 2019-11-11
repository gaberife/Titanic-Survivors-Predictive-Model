import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from collections import Counter

def preprocess(mutatedDF, sourceDF):

    mutatedDF.loc[:, "Sex"] = mutatedDF.loc[:, "Sex"].map(lambda v: 0 if v=="male" else "female")
    
    mutatedDF.loc[:, 'Age'] = mutatedDF.loc[:, 'Age'].fillna(sourceDF.loc[:, 'Age'].median())
    
    mutatedDF.loc[:, "Embarked"] = mutatedDF.loc[:, "Embarked"].map(
            lambda v: 0 if v=="C" else (1 if v=="Q" else (2 if v=="S" else v )))
	    
    mutatedDF.loc[:, "Embarked"] = mutatedDF.apply(lambda row: mutatedDF.loc[:, "Embarked"].mode() 
    if np.isnan(row['Embarked']) else row['Embarked'], axis=1)
	
    mutatedDF.loc[:, 'Fare'] = mutatedDF.loc[:, 'Fare'].fillna(sourceDF.loc[:, 'Fare'].median())
   
def buildAndTestModel():
    titanicTrainSet = pd.read_csv("data/train.csv")
    preprocess(titanicTrainSet, titanicTrainSet)
    titanicTestSet = pd.read_csv("data/test.csv")
    preprocess(titanicTestSet, titanicTrainSet)	    
    
    
    predictorsTrainDF = titanicTrainSet.loc[:, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    survivalTrainDF = titanicTrainSet.loc[:, 'Survived']

    logReg = LogisticRegression(random_state=0, solver='liblinear').fit(predictorsTrainDF, survivalTrainDF)
    print(model_selection.cross_val_score(logReg, predictorsTrainDF, survivalTrainDF, cv=3).mean())

    predictorsTestSet = titanicTestSet.loc[:, ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']]
    predictions = logReg.predict(predictorsTestSet)
    print(predictions, Counter(predictions), sep="\n")
    submitDF = pd.DataFrame(
                    {"PassengerId": titanicTestSet.loc[:,'PassengerId'],
                     "Survived": predictions
                     }
                )
    submitDF.to_csv("data/submitPredictions.csv", index=False)
     
