import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier



#######################################################
# Loading the dataset as dataframe using pandas
#######################################################

dataframe = pd.read_csv('/content/sample_data/Income Dataset (50k).csv')
dataframe.head(3)

dataframe.shape



#######################################################
# Handling missing values 
#######################################################

dataframe.isnull().sum()

# droping rows where workclass is null
dataframe['workclass'].unique()

dataframe = dataframe.dropna(axis = 0, subset = ['workclass'])

dataframe.isnull().sum()
dataframe['workclass'].unique()

# droping rows where native-country is null
dataframe['native-country'].unique()

dataframe = dataframe.dropna(axis = 0, subset = ['native-country'])

dataframe.isnull().sum()
dataframe['native-country'].unique()

# puting 'Other-service' in null positions of occupation column
dataframe['occupation'].fillna("Other-service", inplace = True)



#######################################################
# Encoding categorical features
#######################################################

dataframe['gender'].unique()

# Apply binary encoding to the "gender" column
dataframe['gender_enc'] = LabelEncoder().fit_transform(dataframe['gender'])

# Compare the two columns
dataframe[['gender', 'gender_enc']].head(8)

# Applying one hot encoding

# all categorical features
cat_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']

# appplying one-hot-encoding for every categorical features
encoded_DataFrame= pd.get_dummies(dataframe, columns = cat_features)



#######################################################
# Split the dataset into features and labels
#######################################################

# splitting dataset to test and train

drop = ['gender']

X = encoded_DataFrame.drop(drop, axis=1)
Y = encoded_DataFrame['income_>50K']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=1)

X_train.shape
X_test.shape



#######################################################
# Scaling all the values between 0-1
#######################################################

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)

#print("per-feature minimum before scaling:\n {}".format(X_train.min(axis=0)))
#print("per-feature maximum before scaling:\n {}".format(X_train.max(axis=0)))

#print("per-feature minimum after scaling:\n {}".format(X_train_scaled.min(axis=0)))
#print("per-feature maximum after scaling:\n {}".format(X_train_scaled.max(axis=0)))



#######################################################
# Applying ML
#######################################################

knn=KNeighborsClassifier()

# train
knn.fit(X_train_scaled, y_train)

# scoring on the scaled test set
print("Scaled test set accuracy: {:.2f}".format(knn.score(X_test_scaled, y_test)))