#Importing required libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt  
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA as pca
from factor_analyzer import FactorAnalyzer
from sklearn.feature_selection import VarianceThreshold, RFE
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import classification_report, accuracy_score, classification_report, confusion_matrix
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, roc_curve, plot_roc_curve
from sklearn.feature_extraction.image import grid_to_graph
from sklearn import tree
import seaborn as sns  
from sklearn.neural_network import MLPClassifier
from imblearn.over_sampling import SMOTENC
#For displaying the tree
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO
from IPython.display import Image, display
import pydotplus
import plotly.graph_objs as go
import plotly.io as pio
import plotly

#Importing the data
data= pd.read_csv('/content/drive/My Drive/Company Project/churn_data.csv')

#Converting to respective datatypes
data['gender'] = data['gender'].astype('category', copy = False)
data['Partner'] = data['Partner'].astype('category', copy = False)
data['Dependents'] = data['Dependents'].astype('category', copy = False)
data['PhoneService'] = data['PhoneService'].astype('category', copy = False)
data['MultipleLines'] = data['MultipleLines'].astype('category', copy = False)
data['InternetService'] = data['InternetService'].astype('category', copy = False)
data['OnlineSecurity'] = data['OnlineSecurity'].astype('category', copy = False)
data['OnlineBackup'] = data['OnlineBackup'].astype('category', copy = False)
data['DeviceProtection'] = data['DeviceProtection'].astype('category', copy = False)
data['TechSupport'] = data['TechSupport'].astype('category', copy = False)
data['StreamingTV'] = data['StreamingTV'].astype('category', copy = False)
data['StreamingMovies'] = data['StreamingMovies'].astype('category', copy = False)
data['Contract'] = data['Contract'].astype('category', copy = False)
data['PaperlessBilling'] = data['PaperlessBilling'].astype('category', copy = False)
data['PaymentMethod'] = data['PaymentMethod'].astype('category', copy = False)
data['Churn'] = data['Churn'].astype('category', copy = False)

data['TotalCharges'] = data['TotalCharges'].apply(pd.to_numeric, errors='coerce') 
data['InternetService'].unique()

#Histogram of Tenure
plt.hist(data['tenure'])

#Boxplot of Tenure
plt.boxplot(data['tenure'])

#Replacing missing values in the variables totol charges to 0
data['TotalCharges'].fillna(0, inplace=True)
plt.hist(data['TotalCharges'])

#Small modifications in the data
data[data.MultipleLines == 'No phone service']
data.loc[data['MultipleLines'] == 'No phone service', 'MultipleLines'] = 'No'
data.loc[data['OnlineSecurity'] == 'No internet service', 'OnlineSecurity'] = 'No'
data.loc[data['OnlineBackup'] == 'No internet service', 'OnlineBackup'] = 'No'
data.loc[data['DeviceProtection'] == 'No internet service', 'DeviceProtection'] = 'No'
data.loc[data['TechSupport'] == 'No internet service', 'TechSupport'] = 'No'
data.loc[data['StreamingTV'] == 'No internet service', 'StreamingTV'] = 'No'
data.loc[data['StreamingMovies'] == 'No internet service', 'StreamingMovies'] = 'No'
data.sort_values(by='tenure')

#Binary coding
data.gender.replace(['Male','Female'],[0,1], inplace=True)
binary_cols = ['Partner','Dependents','PhoneService', 'MultipleLines', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'PaperlessBilling']
for i in binary_cols:
    data[i] = data[i].replace({'No': 0, 'Yes':1})

data.Churn.replace(['No','Yes'], ['Not Churn','Churn'],inplace=True)
data['Churn'].value_counts(normalize=True)

#Dummy variable creation
data_dummy1 = pd.get_dummies(data['InternetService'], prefix='IS_')
data_dummy1.head()
data = data.join(data_dummy1)

data_dummy2 = pd.get_dummies(data['Contract'], prefix='Contract_')
data_dummy2.head()
data = data.join(data_dummy2)

data_dummy3 = pd.get_dummies(data['PaymentMethod'], prefix='PM_')
data_dummy3.head()
data = data.join(data_dummy3)
data.dtypes

#Variable standization
data1 = data.loc[:,['tenure','MonthlyCharges','TotalCharges']]
scaler = preprocessing.StandardScaler()
transformed = scaler.fit_transform(data1)
transformed = pd.DataFrame(transformed)
data = data.join(transformed)
data = data.rename(columns={0: 'tenure_T', 1: 'MonthlyCharges_T', 2: 'TotalCharges_T'})

data_final = data.copy()
data_final = data_final.drop(['InternetService', 'Contract', 'PaymentMethod', 'tenure', 'MonthlyCharges', 'TotalCharges', 'customerID'], axis=1)
data_final.columns

#Moving the target variable to the end
cols = list(data_final.columns.values) #List of all of the columns
cols.pop(cols.index('Churn')) #Remove price from list
data_final = data_final[cols+['Churn']]

#Obtain eigenvalues
data_final1 = data_final.drop(['Churn'], axis = 1)
pca_result = pca(n_components=5).fit(data_final1)
pca_result.explained_variance_

#Components from the PCA
pca_result.components_.T * np.sqrt(pca_result.explained_variance_)

#Variance explained by PCA
plt.figure(figsize=(7,5))
plt.plot([1,2,3,4,5], pca_result.explained_variance_ratio_, '-o')
plt.ylabel('Proportion of Variance Explained') 
plt.xlabel('Principal Component') 
plt.xlim(0.75,4.25) 
plt.ylim(0,0.5) 
plt.xticks([1,2,3,4,5])

#Factor Analysis
fa = FactorAnalyzer()
x1 = data_final1.astype(object)
fa.analyze(x1, 3, rotation='varimax')
fa.loadings
x1['tenure_T'] = x1['tenure_T'].astype(float)
x1['MonthlyCharges_T'] = x1['MonthlyCharges_T'].astype(float)
x1['TotalCharges_T'] = x1['TotalCharges_T'].astype(float)
x1.isnull().sum()
x1.dtypes
x1.iloc[:,0:23] = x1.iloc[:,0:23].astype(int)
np.isfinite(x1).sum()

fa.analyze(x1, 5, rotation='varimax')
fa.loadings

# Removing Correlated features
corr_features = set()
# Separating X and y
X = data_final.drop('Churn', axis=1)
y = data_final[['Churn']]
corr_matrix = X.corr()

for i in range(len(corr_matrix.columns)):
    for j in range(i):
        if abs(corr_matrix.iloc[i, j]) > 0.8:
            colname = corr_matrix.columns[i]
            corr_features.add(colname)
            if colname in data_final.columns:
                    del data_final[colname]
            if colname in X.columns:
                    del X[colname]


y = y.astype('category', copy = False)
X.iloc[:,0:23] = X.iloc[:,0:23].astype('category', copy = False)

#SMOTE sampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)


target = ['Churn','Not Churn']
freq = [2409,3614]
sampling = dict(zip(target,freq))
print(sampling)

smt = SMOTENC(random_state=42,categorical_features=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23], sampling_strategy=sampling)
X_train1, y_train1 = smt.fit_resample(X_train, y_train)
X_train = pd.DataFrame(X_train1, columns=X_train.columns)
y_train = pd.Series(y_train1)

#Converting objects to categorical variables
X_train['gender'] = X_train['gender'].astype('category', copy = False)
X_train.iloc[:,0:23] = X_train.iloc[:,0:23].astype('category')
X_test.iloc[:,0:23] = X_test.iloc[:,0:23].astype('category', copy = False)

values, counts = np.unique(y_train, return_counts=True)
counts

# Performing Recursive Feature elimination
logreg = LogisticRegression()
rfe = RFE(logreg, 10, step=1)
rfe = rfe.fit(X_train, y_train.values.ravel())
print(rfe.support_)
print(rfe.ranking_)
features_selected = rfe.get_support(1) #the most important features
X_train_lr = X_train[X_train.columns[features_selected]] # final features`
X_test_lr = X_test[X_test.columns[features_selected]] 

X_train_lr.columns

# Logistic Regression Classifier model
logreg = LogisticRegression()
logreg.fit(X_train_lr, y_train)
y_pred = logreg.predict(X_test_lr)

#Confusion matrix
confusion_matrix_lr = confusion_matrix(y_test, y_pred)
print(confusion_matrix_lr)

#Evaluation Metrics
print('Accuracy of Logistic Regression classifier on test set: {:.3f}'.format(accuracy_score(y_test, y_pred)))
print('Classification report of Logistic Regression:\n', classification_report(y_test, y_pred))

clf = LogisticRegression().fit(X_train_lr, y_train)
print(clf.coef_, clf.intercept_)

# The below values are from clf.coef_
values1 = [ 0.87794826,  0.77451129,  0.97895797,  1.9781441 ,  1.28622139,
         1.38411178,  0.80588321,  1.25078858,  0.89129903, -0.97918123]

#Variable Importance
index = np.arange(len(X_train_lr.columns))
plt.bar(index, values1)
plt.xlabel('Variable', fontsize=10)
plt.ylabel('Coefficient', fontsize=10)
plt.xticks(index, X_train_lr.columns, fontsize=10, rotation=60)
plt.title('Coefficient Importance')
plt.show()

#Heat map of confusion matrix
ax= plt.subplot()
sns.heatmap(confusion_matrix_lr, annot=True, ax = ax)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix of Logistic Regression Classifier')
ax.xaxis.set_ticklabels(['Churn', 'Not Churn'])
ax.yaxis.set_ticklabels(['Churn', 'Not Churn'])

#Classnames and column names
col_names = list(data_final.columns.values)
classnames = list(data_final.Churn.unique())

#Decision Tree Model
tre2 = tree.DecisionTreeClassifier().fit(X_train, y_train)

# Tree Visualization
dot_data = StringIO()
tree.export_graphviz(tre2, out_file=dot_data,
                     feature_names=col_names[0:25],
                     class_names=classnames,
                     max_depth = 3,
                     filled=True,
                     rounded=True, 
                     special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
display(Image(graph.create_png()))

#Evaluation Metric of Decision Tree
clf = tree.DecisionTreeClassifier(max_depth = 5, random_state = 0)
tre2 = clf.fit(X_train, y_train)
predicted = tre2.predict(X_test)
cm_decisiont = metrics.confusion_matrix(y_test, predicted)
print('Confusion Matrix of Decision Tree classifier on test set:\n')
print(cm_decisiont)
accuracy_decision = clf.score(X_test, y_test)
print('Accuracy of Decision Tree classifier on test set: {:.3f}', (round(accuracy_decision,3)))

print('\n Classification report of Decision Tree classifier:\n', metrics.classification_report(y_test, predicted))


# Feature importance for Decision Tree model
importances_decision_tree = pd.DataFrame({'feature':X_train.columns,'importance':np.round(clf.feature_importances_,3)})
importances_decision_tree = importances_decision_tree.sort_values('importance',ascending=False)
print(importances_decision_tree)

# List to store the average RMSE for each value of max_depth:
max_depth_range = list(range(1, 10))
accuracy = [] 
for depth in max_depth_range:
    clf = tree.DecisionTreeClassifier(max_depth = depth, 
                             random_state = 0)
    clf.fit(X_train, y_train)    
    score = clf.score(X_test, y_test)
    accuracy.append(score)

#Variable Importance Bar Chart
importances_decision_tree[:15].plot(x="feature", y="importance", kind="bar")

#Heat map of confusion matrix from Decision Tree
ax= plt.subplot()
sns.heatmap(cm_decisiont, annot=True, ax = ax)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix of Decision Tree Classifier')
ax.xaxis.set_ticklabels(['Churn', 'Not Churn'])
ax.yaxis.set_ticklabels(['Churn', 'Not Churn'])

# Neural Network   
nnclass2 = MLPClassifier(activation='relu', solver='sgd', hidden_layer_sizes=(100,100))
nnclass2.fit(X_train, y_train)
nnclass2_pred = nnclass2.predict(X_test)

#Evaluation Metrics
cm_neural = metrics.confusion_matrix(y_test, nnclass2_pred)
print('Confusion Matrix of Neural Net classifier on test set:\n')
print(cm_neural)
accuracy_decision = nnclass2.score(X_test, y_test)
print('\n Accuracy of Neural Net on test set: {:.3f}', (round(accuracy_decision,3)))

print('\n Classification report of Neural Network:\n', metrics.classification_report(y_test, nnclass2_pred))

# Confusion Matrix for Neural Network model
ax= plt.subplot()
sns.heatmap(cm_neural, annot=True, ax = ax)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_title('Confusion Matrix of Neural Network')
ax.xaxis.set_ticklabels(['Churn', 'Not Churn'])
ax.yaxis.set_ticklabels(['Churn', 'Not Churn'])

#Comparision of Roc Curves
disp = plot_roc_curve(logreg, X_test_lr, y_test)
plot_roc_curve(clf, X_test, y_test, ax=disp.ax_);
plot_roc_curve(nnclass2, X_test, y_test, ax=disp.ax_);

#Distribution of Target variable
y_train.describe()
y_test.describe()

#Distribution of Continuous variables
X_train.describe()
X_test.describe()

#Distribution of Categorical variables
X_train.iloc[:,0:23].describe()
X_test.iloc[:,0:23].describe()
