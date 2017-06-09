from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import KFold
import numpy as np
import pandas as pd

# Import and clean the data
titanic = pd.read_csv('data/train.csv')

titanic['Age'] = titanic['Age'].fillna(titanic['Age'].median())

titanic.loc[titanic['Sex'] == 'male', 'Sex'] = 0
titanic.loc[titanic['Sex'] == 'female', 'Sex'] = 1

titanic.loc[titanic['Embarked'] == 'S', 'Embarked'] = 0
titanic.loc[titanic['Embarked'] == 'C', 'Embarked'] = 1
titanic.loc[titanic['Embarked'] == 'Q', 'Embarked'] = 2
titanic['Embarked'] = titanic['Embarked'].fillna(titanic['Embarked'].median())

# Find nan in Embarked field
# print(titanic.loc[titanic['Embarked'].isnull()])
# print(titanic.loc[titanic['Embarked'] == 0, 'Embarked'].describe())
# print(titanic.loc[titanic['Embarked'] == 1, 'Embarked'].describe())
# print(titanic.loc[titanic['Embarked'] == 2, 'Embarked'].describe())

# Predict
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
#titanic[predictors].isnull())
alg = LinearRegression()
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    train_predictors = (titanic[predictors].iloc[train, :])
    train_target = titanic['Survived'].iloc[train]

    alg.fit(train_predictors, train_target)

    test_predictions = alg.predict(titanic[predictors].iloc[test, :])

    predictions.append(test_predictions)

predictions = np.concatenate(predictions, axis=0)
predictions[predictions > .5] = int(1)
predictions[predictions <=.5] = int(0)
predictions.astype(int)
print(predictions)

accuracy = sum(predictions[predictions == titanic["Survived"]]) / len(predictions)

# print(sum(predictions[predictions == titanic["Survived"]]))
# print(len(predictions))
print(accuracy)
