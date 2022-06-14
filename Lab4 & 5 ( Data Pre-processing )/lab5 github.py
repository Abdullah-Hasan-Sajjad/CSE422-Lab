import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier


dataframe = pd.read_csv('/content/sample_data/Income Dataset (50k).csv')

# getting null positions and drop those rows
#print(dataframe.isnull().sum())
dataframe = dataframe.dropna(axis = 0, subset = ['workclass','occupation','native-country'])


# Encoding categorical features

# Apply binary encoding to the "gender" column
dataframe['gender_enc'] = LabelEncoder().fit_transform(dataframe['gender'])

# all categorical features
cat_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']

# appplying one-hot-encoding for every categorical features
encoded_DataFrame= pd.get_dummies(dataframe, columns = cat_features)

# splitting dataset to test and train
drop = ['gender','income_>50K']
X = encoded_DataFrame.drop(drop, axis=1)
Y = encoded_DataFrame['income_>50K']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=1)

# Scaling all the values between 0-1
scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

# applying ML
knn=KNeighborsClassifier()

# train
knn.fit(X_train_scaled, y_train)

# scoring on the scaled test set
print("Scaled test set accuracy: {:.2f}".format(knn.score(X_test_scaled, y_test)))