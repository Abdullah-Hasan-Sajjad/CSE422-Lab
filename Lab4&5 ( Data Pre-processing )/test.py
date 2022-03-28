import pandas as pd
import numpy as np

#loading csv file
dataframe = pd.read_csv('Income Dataset (50k).csv')
dataframe.head(3)

dataframe = dataframe.dropna(axis = 0, subset = ['workclass'])
dataframe = dataframe.dropna(axis = 0, subset = ['native-country'])
dataframe['occupation'].fillna("Other-service", inplace = True)
#print(dataframe.isnull().sum())

from sklearn.preprocessing import LabelEncoder

# Set up the LabelEncoder object
enc = LabelEncoder()

# Apply the encoding to the 'gender' column
dataframe['gender_enc'] = enc.fit_transform(dataframe['gender'])

# splitig data frame
from sklearn.model_selection import train_test_split

# selecting feature columns
drop = ['gender','income_>50K']
X = dataframe.drop(drop, axis=1)

Y = dataframe['income_>50K']


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=1)
#print(X_train.shape)
#print(X_test.shape)

# one-hot-encoding categorical features
from feature_engine.encoding import OneHotEncoder

cat_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']

ohe_encoder = OneHotEncoder(variables=cat_features)
ohe_encoder.fit(X_train)

X_ohe_train = ohe_encoder.transform(X_train)
print(X_ohe_train.shape)
print(X_ohe_train.columns)