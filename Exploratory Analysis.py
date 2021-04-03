#Import all the required libraries
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt  
from sklearn import preprocessing

from sklearn.decomposition import PCA as pca
from factor_analyzer import FactorAnalyzer

import plotly.graph_objs as go
import plotly.io as pio
import plotly 

#setting up the input directory
os.chdir('D:\Study\Sem2\MSIS5223\project')
#importing data from the file
data = pd.read_csv('churn_data.csv')
data['Churn'].unique()


#converting the data types of object to category
data['SeniorCitizen'] = data['SeniorCitizen'].astype('category', copy = False)
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

#Convert the object datatype to float
data['TotalCharges'] = data['TotalCharges'].apply(pd.to_numeric, errors='coerce') 
data['InternetService'].unique()
data.dtypes

plt.hist(data['tenure'])
plt.boxplot(data['tenure'])
plt.hist(data['MonthlyCharges'])
plt.boxplot(data['MonthlyCharges'])

#number of values with NaN
data['TotalCharges'].isna().sum()

#imputes the NaN with 0
data['TotalCharges'].fillna(0, inplace=True)
plt.hist(data['TotalCharges'])
plt.boxplot(data['TotalCharges'])

#Converting some row values with 'No'
data[data.MultipleLines == 'No phone service']
data.loc[data['MultipleLines'] == 'No phone service', 'MultipleLines'] = 'No'
data.loc[data['OnlineSecurity'] == 'No internet service', 'OnlineSecurity'] = 'No'
data.loc[data['OnlineBackup'] == 'No internet service', 'OnlineBackup'] = 'No'
data.loc[data['DeviceProtection'] == 'No internet service', 'DeviceProtection'] = 'No'
data.loc[data['TechSupport'] == 'No internet service', 'TechSupport'] = 'No'
data.loc[data['StreamingTV'] == 'No internet service', 'StreamingTV'] = 'No'
data.loc[data['StreamingMovies'] == 'No internet service', 'StreamingMovies'] = 'No'
data.sort_values(by='tenure')

#converted the text columns into (1,0) coding 
#Note: after running this if the variable type as changed then convert them back into categorical using first code
data.gender.replace(['Male','Female'],[0,1], inplace=True)
data.Partner.replace(['No','Yes'],[0,1], inplace=True)
data.Dependents.replace(['No','Yes'],[0,1], inplace=True)
data.PhoneService.replace(['No','Yes'],[0,1], inplace=True)
data.MultipleLines.replace(['No','Yes'],[0,1], inplace=True)
data.OnlineSecurity.replace(['No','Yes'],[0,1], inplace=True)
data.OnlineBackup.replace(['No','Yes'],[0,1], inplace=True)
data.DeviceProtection.replace(['No','Yes'],[0,1], inplace=True)
data.TechSupport.replace(['No','Yes'],[0,1], inplace=True)
data.StreamingTV.replace(['No','Yes'],[0,1], inplace=True)
data.StreamingMovies.replace(['No','Yes'],[0,1], inplace=True)
data.PaperlessBilling.replace(['No','Yes'],[0,1], inplace=True)
data.Churn.replace(['No','Yes'],[0,1], inplace=True)
data.dtypes
#created dummy variables for the variables and joined them back to the table
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

#Standardized the variables
data1 = data.loc[:,['tenure','MonthlyCharges','TotalCharges']]
scaler = preprocessing.StandardScaler()
transformed = scaler.fit_transform(data1)
transformed = pd.DataFrame(transformed)
data = data.join(transformed)
data = data.rename(columns={0: 'tenure_T', 1: 'MonthlyCharges_T', 2: 'TotalCharges_T'})

data['customerID'] = data.customerID.str.replace('-', '')

#data_final is the final dataset after removing the untransformed columns
data_final = data
data_final = data_final.drop(['InternetService', 'Contract', 'PaymentMethod', 'tenure', 'MonthlyCharges', 'TotalCharges'], axis=1)
data_final.columns
data_final.shape

# Creating ID and target variables
Id = ['customerID']
target_churn = ['Churn']

# Features
#x = data_final.loc[:, data_final.columns != ['Churn', 'customerID']]

x1 = data_final[[i for i in data_final if i not in Id_col + target_col]]

# Target
Y1 = data_final[target_churn + Id]

# PCA with 10 components
pca = PCA(n_components=10)

principalComponents = pca.fit_transform(x1)
principalDf = pd.DataFrame(data = principalComponents,
             columns = ['PCA1', 'PCA2', 'PCA3', 'PCA4', 'PCA5','PCA6', 'PCA7', 'PCA8', 'PCA9', 'PCA10'])
principalDf = pd.merge(principalDf, Y1,left_index = True, right_index = True, how = 'left')

principalDf.columns

# Variance explained by PCA
pca.explained_variance_ratio_

# PCA visualization PC1 vs PC2
plt.style.use('fivethirtyeight')
plt.tight_layout()
plt.grid = True
def df_pca(churn, col):
    df_scatter = go.Scatter(x = principalDf[principalDf.Churn == churn]['PCA1'],
                        y = principalDf[principalDf.Churn == churn]['PCA2'],
                        text = ('Customer:' + 
                                principalDf[principalDf.Churn == churn]['customerID']),
                                                name = churn, mode = 'markers',
                        marker = {'color' : col,
                                      'line' : {'width' : .7}})
    return df_scatter

def graph_layout():

    layout_pca = go.Layout({'title' : 'Visualizing first two Principal Components',
                        'plot_bgcolor'  : '#FAFAFA',
                        'xaxis' : {'title' : 'Principal Component1'},
                        'yaxis' : {'title' : 'Principal Component2'},
                        'height' : 700
                       })
    return layout_pca

pc1 = df_pca(1,'blue')
pc2 = df_pca(0,'green')
data = [pc1, pc2]
fig = go.Figure(data = data,layout = graph_layout())
plotly.offline.plot(fig, filename = 'pca' + '.html') 



#PCA and Factor Analysis

data_final1 = data_final.drop(['customerID', 'Churn'], axis = 1)
data_final1.dtypes

#Obtain eigenvalues
pca_result.explained_variance_

#Components from the PCA
pca_result.components_.T * np.sqrt(pca_result.explained_variance_)

plt.figure(figsize=(7,5))
plt.plot([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15], pca_result.explained_variance_ratio_, '-o')
plt.ylabel('Proportion of Variance Explained') 
plt.xlabel('Principal Component') 
plt.xlim(0.75,4.25) 
plt.ylim(0,0.5) 
plt.xticks([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15])

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

























