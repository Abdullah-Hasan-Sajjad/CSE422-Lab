import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.decomposition import PCA 



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

drop = ['gender','income_>50K']

X = encoded_DataFrame.drop(drop, axis=1)
Y = encoded_DataFrame['income_>50K']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=1)


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
# Applying Support Vector Machine
#######################################################

svc = SVC(kernel="linear")

# train
svc.fit(X_train_scaled, y_train)

# scoring on the scaled test set
print("Training accuracy of the model is {:.2f}".format(svc.score(X_train_scaled, y_train)))
svc_score_before_PCA=svc.score(X_test_scaled, y_test)
print("Testing accuracy of the model is {:.2f}".format(svc_score_before_PCA))



#######################################################
# Applying  Neural Network (MLPClassifier)
#######################################################

from sklearn.neural_network import MLPClassifier

nnc=MLPClassifier(hidden_layer_sizes=(7), activation="relu", max_iter=10000)

nnc.fit(X_train_scaled, y_train)

print("The Training accuracy of the model is {:.2f}".format(nnc.score(X_train_scaled, y_train)))
neuralNetwork_score_before_PCA=nnc.score(X_test_scaled, y_test)
print("The Testing accuracy of the model is {:.2f}".format(neuralNetwork_score_before_PCA))



#######################################################
# Applying  Random Forest
#######################################################

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=50)

rfc.fit(X_train_scaled, y_train)

print("The Training accuracy of the model is {:.2f}".format(rfc.score(X_train_scaled, y_train)))
randomForest_score_before_PCA=rfc.score(X_test_scaled, y_test)
print("The Testing accuracy of the model is {:.2f}".format(randomForest_score_before_PCA))



#######################################################
# Applying  PCA
#######################################################

scaler = MinMaxScaler()

encoded_scaled_DataFrame = scaler.fit_transform(encoded_DataFrame.drop(['gender'], axis=1))

encoded_scaled_DataFrame.shape

# Dataframe has 104 except the target column 
# so it would be redeuced to 52 columns

pca = PCA(n_components=52)

principal_components= pca.fit_transform(encoded_scaled_DataFrame)
print(principal_components)
pca.explained_variance_ratio_
sum(pca.explained_variance_ratio_)
principal_df = pd.DataFrame(data=principal_components)
principal_df.shape



#######################################################
# Split the dataset into features and labels
#######################################################

# splitting dataset to test and train

X = principal_df
Y = encoded_DataFrame['income_>50K']

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.20, random_state=1)



#######################################################
# Applying Support Vector Machine
#######################################################

svc = SVC(kernel="linear")

# train
svc.fit(X_train_scaled, y_train)

# scoring on the scaled test set
print("Training accuracy of the model is {:.2f}".format(svc.score(X_train_scaled, y_train)))
svc_score_after_PCA=svc.score(X_test_scaled, y_test)
print("Testing accuracy of the model is {:.2f}".format(svc_score_after_PCA))



#######################################################
# Applying  Neural Network (MLPClassifier)
#######################################################

from sklearn.neural_network import MLPClassifier

nnc=MLPClassifier(hidden_layer_sizes=(7), activation="relu", max_iter=10000)

nnc.fit(X_train, y_train)

print("The Training accuracy of the model is {:.2f}".format(nnc.score(X_train, y_train)))
neural_network_score_after_PCA=nnc.score(X_test, y_test)
print("The Testing accuracy of the model is {:.2f}".format(neural_network_score_after_PCA))



#######################################################
# Applying  Random Forest
#######################################################

from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=50)

rfc.fit(X_train, y_train)

print("The Training accuracy of the model is {:.2f}".format(rfc.score(X_train, y_train)))
randomForest_score_after_PCA=rfc.score(X_test, y_test)
print("The Testing accuracy of the model is {:.2f}".format(randomForest_score_after_PCA))



#######################################################
# Ploting bar chart
#######################################################

fig = plt.figure()
ax = fig.add_axes([1,1,2.5,1])

classifiers = ['SVC before PCA','SVC after PCA', 'Neural Network before PCA', 'Neural Network after PCA','Random Forest before PCA','Random Forest after PCA']
scores = [svc_score_before_PCA,svc_score_after_PCA,neuralNetwork_score_before_PCA,neural_network_score_after_PCA,randomForest_score_before_PCA,randomForest_score_after_PCA]
ax.bar(classifiers,scores,color ='yellow')

plt.show()
