import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score



#######################################################
# Loading the dataset as dataframe using pandas
#######################################################

dataframe = pd.read_csv('/content/sample_data/Income Dataset (50k).csv')
dataframe.head(3)

dataframe.shape



#######################################################
# Handling missing values 
#######################################################

# droping rows where workclass and native-country is null
dataframe = dataframe.dropna(axis = 0, subset = ['workclass','native-country'])

# puting 'Other-service' in null positions of occupation column
dataframe['occupation'].fillna("Other-service", inplace = True)



#######################################################
# Encoding categorical features
#######################################################

# Apply binary encoding to the "gender" column
dataframe['gender_enc'] = LabelEncoder().fit_transform(dataframe['gender'])


# Applying one hot encoding

# all categorical features
cat_features = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country']

# appplying one-hot-encoding for every categorical features
encoded_DataFrame= pd.get_dummies(dataframe, columns = cat_features)



#######################################################
# Split the dataset into features and labels
#######################################################

# splitting dataset to test and train

drop = ['gender','income_>50K']

X = encoded_DataFrame.drop(drop, axis=1)
Y = encoded_DataFrame['income_>50K']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=42)



#######################################################
# Scaling all the values between 0-1
#######################################################

scaler = MinMaxScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.fit_transform(X_test)



#######################################################
# Applying LogisticRegression
#######################################################

model = LogisticRegression()

model.fit(X_train_scaled, y_train) #Training the model

predictions = model.predict(X_test_scaled)

LogisticRegression_score=accuracy_score(y_test, predictions)

print(LogisticRegression_score)



#######################################################
# Applying DecisionTreeClassifier
#######################################################

clf = DecisionTreeClassifier(criterion='entropy',random_state=1)

clf.fit(X_train_scaled, y_train)

y_pred = clf.predict(X_test_scaled)

decisionTree_score=accuracy_score(y_pred,y_test)

print(decisionTree_score)



#######################################################
# Ploting bar chart
#######################################################

fig = plt.figure()
ax = fig.add_axes([0,0,1,1])

classifiers = ['Logistic Regression Score', 'Decision Tree Score']
scores = [LogisticRegression_score,decisionTree_score]
ax.bar(classifiers,scores,color ='yellow')

plt.show()