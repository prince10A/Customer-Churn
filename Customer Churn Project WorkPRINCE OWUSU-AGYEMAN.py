#!/usr/bin/env python
# coding: utf-8

# ### Student Name- Prince Owusu-Agyeman

# # Customer Churn Prediction

# ## Import  Libraries 
# 

# In[1]:


# Data Analysis and Visualization Libraries
import numpy as np
import pandas as pd
from scipy.stats import skew
import matplotlib.pyplot as plt
import seaborn as sns

# Data Preprocessing Libraries
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

# Classification and Evaluation Libraries 
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC

from collections import Counter  #counting

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

import warnings
warnings.filterwarnings('ignore')


# In[ ]:





# In[ ]:





# In[166]:


# import the  dataset

df= pd.read_csv('Customer-Churn.csv')
df.head()


# In[ ]:





# In[87]:


df.shape


# In[ ]:





# In[89]:


df.info()


# In[ ]:





# In[90]:


# Statistical Description of the Numerical Features. 
df.describe()


# In[6]:


# Statistical Description of the Categorical  Features.


# In[91]:


df.describe(include='object')


# In[ ]:





# In[92]:


# This is to know the number of customers who churned and those who did not. 
df["Churn"].value_counts()


# In[94]:


ax=sns.countplot(x=df["Churn"], order=df["Churn"].value_counts(ascending=False).index)
values=df["Churn"].value_counts(ascending=False).values
ax.bar_label(container=ax.containers[0], labels=values);


# In[93]:


#check for duplicated values

dup = df.duplicated().sum()
print("Number of duplicates:", dup)


# In[10]:


#check for missing data

df_churn.isnull().sum()


# In[99]:


plt.figure(figsize=(10,4))
sns.heatmap(df.isnull(), cbar=True, cmap="Blues_r");


# In[95]:


#Convert total charges to a float.

df.TotalCharges = pd.to_numeric(df.TotalCharges, errors= "coerce")
df.isnull().sum()


# ##### 11 values missing for Total Charges 

# In[ ]:





# In[ ]:





# ### Data Cleaning

# In[103]:


# removal of rows with missing data

df.dropna(inplace=True)


# In[104]:


# Updating wrongly labeled data points: some data cells has 'No phone service' and 'No internet service' intead of 'No' 

column_update = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                 'StreamingMovies']

# Looping through columns and replace 'No phone service' and 'No internet service' with 'No'
for c in column_update:
    df[c] = df[c].replace('No phone service', 'No')
    df[c] = df[c].replace('No internet service', 'No')


# In[105]:


df


# ###### Grouping the "tenure" into 1-12 months, 1-2 years and 2-3 years, etc
# 
# 

# In[107]:


# get the maximum tenure.  #72
print(df[ "tenure" ].max())


# In[108]:


#grouping tenure into groups of 12 months. 

labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
df["tenure_group"] = pd.cut(df.tenure, range(1, 80, 12), right=False, labels=labels)


# In[109]:


df["tenure_group"].value_counts()


# In[110]:


df.head()


# ## Exploratory Data Analysis 

# ### Univariate Analysis

# In[115]:


#I can equally use this syntax and explain the insights of the graphs I will get. 
for i, predictor in enumerate(df.drop(columns=['Churn', 'TotalCharges', 'MonthlyCharges'])):
    plt.figure(i)
    sns.countplot(data=df, x=predictor, hue='Churn')


# In[ ]:





# In[116]:


# Distribution of customer tenure group
plt.figure(figsize=(12,6))
sns.histplot(df['tenure_group'], edgecolor='yellow', bins=20, kde=True)
plt.title('Customer Tenure Group Distribution')
plt.xlabel('Months')
plt.ylabel('Count');


# ##### Insights : the customer group from 1 month to a year  and those from 61-72 months (6 years) form the the hightest number of customers in connecTel.

# ### Churn by monthly charges and total charges

# In[118]:


Mth = sns.kdeplot(df.MonthlyCharges[(df["Churn"]==0)],
                  color="Red", shade=True)
Mth = sns.kdeplot(df.MonthlyCharges[(df["Churn"]==1)],
                 ax=Mth, color="Blue", shade= True)
Mth.legend(["No Churn", "Churn"], loc='upper right')
Mth.set_ylabel('Density')
Mth.set_xlabel('Monthly Charges')
Mth.set_title('Monthly Charges by Churn')


# In[119]:


Tot = sns.kdeplot(df.MonthlyCharges[(df["Churn"]==0)],
                  color="Red", shade=True)
Tot = sns.kdeplot(df.MonthlyCharges[(df["Churn"]==1)],
                 ax=Tot, color="Blue", shade= True)
Tot.legend(["No Churn", "Churn"], loc='upper right')
Tot.set_ylabel('Density')
Tot.set_xlabel('Monthly Charges')
Tot.set_title('Monthly Charges by Churn')


# ### a higher churn at lower total charges
# ### however if we combine the insights of 3 paramenters ie. Tenure, monthly charges and total charges then the picture is a bit clearer
# ### therefore higher monthly charge at lower tenure results into lower total charge. hence all these 3 factors viz higher monthly¶
# ### monthly charge, lower tenure and lower total charge are linked to high churn
# ### build a correlation of all predictors with churn

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[24]:


# Customers gender distribution
plt.figure(figsize=(7,7))
plt.pie(df_churn['gender'].value_counts(), labels=df_churn['gender'].value_counts().index, autopct='%1.2f%%')
plt.title('ConnectTel Customers Gender Distribution');


# #### There is a gender balance among the customers only 0.96% difference between both gender

# In[ ]:





# In[126]:


# Monthly charges distribution
plt.figure(figsize=(10,6))
sns.histplot(df_churn['MonthlyCharges'], edgecolor='red', bins=20, kde=True)
plt.title('Monthly Charges Distribution')
plt.xlabel('Monthly Charge')
plt.ylabel('Count');

# Calculate the skewness of the MonthlyCharges
monthly_charges_skewness = skew(df_churn['MonthlyCharges'])
print(f"Skewness of MonthlyCharges: {monthly_charges_skewness:.2f}")


# ####  Most customers, paid monthly charges of 20usd however the next majority also went for the 80 usd montly plan package.

# In[ ]:





# In[125]:


# Total charges distribution
plt.figure(figsize=(10,6))
sns.histplot(df_churn['TotalCharges'], edgecolor='red', bins=20, kde=True)
plt.title('Total Charges Distribution')
plt.xlabel('Total Charge')
plt.ylabel('Count');


# ##### From the histogram, the Total Charges  from customers are positively skewed with a large proportion of customers bringing in total revenue less than 2000 USD.  We can conclude that the customers who are bringing in hign revenues are few.

# In[ ]:





# In[127]:


# ConnecTel customer churn
plt.figure(figsize=(10,5))
ax = sns.countplot(data=df_churn, x='Churn', edgecolor='blue')
plt.title('ConnecTel Customer Churn Rate')
plt.xticks(ticks=[0, 1], labels=['No Churn', 'Churn']);

# Percentage calculation
total = float(len(df_churn['Churn']))
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height() / total)
    x = p.get_x() + p.get_width() / 2 - 0.1
    y = p.get_y() + p.get_height()
    ax.annotate(percentage, (x, y), size=15) 


# #####  Customers have a no churn rate of 73.5% and a churn rate of 26.5%

# In[130]:


df['Contract'].value_counts().plot(kind='pie',title='Contract', autopct='%1.0f%%')


# ##### From the pie chart, it is evident that the highest contract type is month to month, where as the two year contract follows and then the one year contract. 

# In[ ]:





# In[29]:


#Relationship between Monthly Charges and Total Charges

sns.lmplot(data=df_churn, x="MonthlyCharges", y="TotalCharges",  fit_reg=False, )


# ##### We can simply see that the more tthe monthly charges the more total charges also increases. 

# In[ ]:





# In[ ]:





# In[ ]:





# ### Bivariate Analysis

# In[ ]:





# In[160]:


# Customer churn by Tenure
plt.figure(figsize=(10 ,6))
sns.barplot(data=df, x='Churn', y='tenure', edgecolor='red')
plt.xlabel('Churn')
plt.ylabel('Tenure')
plt.title('Customer Churn by Tenure')
plt.xticks(ticks=[0, 1], labels=['No Churn', 'Churn']);


# #### We can see from here that, the new customers (1-16) months, which was about 50% churned as compared to the customers who had been with ConnecTel till about 36 months. 

# In[ ]:





# In[162]:


# Monthly charges distribution by customer churn
plt.figure(figsize=(12,4))
sns.boxplot(data=df, x='Churn', y='MonthlyCharges')
plt.ylabel('Monthly Charges')
plt.title('Monthly Charges by Churn')
plt.xticks(ticks=[0, 1], labels=['No Churn', 'Churn']);

# Mean of monthly charges for churned and retained customers
mean_monthly_charges_churned = df[df['Churn'] == 'Yes']['MonthlyCharges'].mean()
mean_monthly_charges_retained = df[df['Churn'] == 'No']['MonthlyCharges'].mean()
print(f"Mean Monthly Charges for Churned Customers: {mean_monthly_charges_churned:.2f}")
print(f"Mean Monthly Charges for Retained Customers: {mean_monthly_charges_retained:.2f}")


# #### From the boxplot above, we can see that theaverage  monthly charges for retained customers is 61.31, However with customers in the churn category, they pay as much as  an average of  74.44 monthly, making them the customers who bring in the highest revenue. 
# 

# In[ ]:





# In[165]:


#Customer Contract Type by Churn
plt.figure(figsize=(8,6))
sns.countplot(data=df, x='Contract', hue='Churn')
plt.xlabel('Contract')
plt.title('Contract Type by Churn');


# #### Customers who are on the monthly contracts have the highest non churn and the highest number of churn as well. those on the two year contract, also have a high number of customers who have not churned and then those who have churned are very low. Then those on the one year contract have a low churn rate in comparison with customers who did not churn in that contract type. 

# In[ ]:





# In[33]:


#  ConnecTel Services offered to Customers Via  Churn
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(16,12))

#Internet Service by Churn
sns.countplot(data=df_churn, x='InternetService', hue='Churn', ax=axs[0,0])

# Tech Support  by Churn
sns.countplot(data=df_churn, x='TechSupport', hue='Churn', ax=axs[0,1])

# Online Security Service by Churn
sns.countplot(data=df_churn, x='OnlineSecurity', hue='Churn', ax=axs[0,2])

# Online Backup Service by Churn
sns.countplot(data=df_churn, x='OnlineBackup', hue='Churn', ax=axs[1,0])

# Phone Service by Churn
sns.countplot(data=df_churn, x='PhoneService', hue='Churn', ax=axs[1,1])

# Device Protection  Service by Churn
sns.countplot(data=df_churn, x='DeviceProtection', hue='Churn', ax=axs[1,2]);


# In[ ]:





# In[167]:


df['PaymentMethod'].value_counts().plot(kind='pie',title='PaymentMethod', autopct='%1.0f%%')


# #### The electronic check payment method was the most preferred method. 

# In[ ]:





# In[157]:


df.info()


# In[ ]:





# In[ ]:





# In[ ]:





# In[156]:


# Customer Demographics by Churn
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15,10))

# Gender Distribution by Churn
sns.countplot(data=df, x='gender', hue='Churn', ax=axs[0,0])
axs[0,0].set_title('Gender by Churn')

# Citizenship Type by Churn
sns.countplot(data=df, x='SeniorCitizen', hue='Churn', ax=axs[0,1])
axs[0,1].set_title('Citizenship Type by Churn')

# Customer relationship by Churn
sns.countplot(data=df, x='Partner', hue='Churn', ax=axs[1,0])
axs[1,0].set_title('Partnered Customers by Churn')

# Customer dependency by Churn
sns.countplot(data=df, x='Dependents', hue='Churn', ax=axs[1,1])
axs[1,1].set_title('Dependent Customers by Churn');


# In[ ]:





# In[ ]:





# In[155]:


df


# # MACHINE LEARNING

# ## Data Pre Processing

# #### Encoding of categorical columns into numerical

# In[154]:


#method to return columns with object data types (categorical)
def col_unique_val(d):
    for col in d:
        if df[col].dtypes == 'object':
            print(f'{col}: {df[col].unique()}')


# In[169]:


#apply the method on dataframe
col_unique_val(df)


# In[41]:


#create an array of columns to be encoded
le_columns = ['Partner', 'Dependents', 'PhoneService', 'MultipleLines','OnlineSecurity','OnlineBackup',
              'DeviceProtection','TechSupport','StreamingTV', 'StreamingMovies','PaperlessBilling']


# In[170]:


df[le_columns] = df[le_columns].replace({'Yes': 1, 'No' : 0})


# In[171]:


col_unique_val(df)


# In[172]:


df['gender'] = df['gender'].replace({'Male': 1, 'Female' : 0})


# In[173]:


df['gender']


# In[174]:


col_unique_val(df)


# In[47]:


#Perform one-hot encoding on the remaining categorical columns


# In[138]:


df = pd.get_dummies(data=df, columns=['InternetService','Contract','PaymentMethod'])


# In[139]:


pd.set_option('display.max_columns', None)


# In[140]:


df


# In[141]:


#create an instance of labelencoder()
labelEn = LabelEncoder()

#convert churn column into numerical values
df['Churn'] = labelEn.fit_transform(df['Churn'])


# In[142]:


df['Churn']


# In[143]:


df = df.drop('tenure_group', axis=1)


# In[144]:


df


# ## Scaling

# In[56]:


#create an array of the numerical features
numericalCols = ['tenure', 'TotalCharges', 'MonthlyCharges']


# In[145]:


# Scaling of predictor variables

# Initialize scaler and fit-tranform data using scaler
scaler = MinMaxScaler()

df[numericalCols] = scaler.fit_transform(df[numericalCols])


# In[146]:


df


# ### DATA SPLITTING

# In[59]:


#Split data into x and y 


# In[147]:


x = df.drop('Churn', axis=1)
y = df['Churn']


# In[148]:


x


# In[149]:


y


# In[176]:


# Split the  data into training and testing sets

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


# In[177]:


X_train


# In[152]:


y_train


# In[ ]:





# In[ ]:





# ###  Model Building

# #Algorithms for model:
# Logistic Regression
# Naive Bayes
# Random Forest
# Support Vector Machines

# In[175]:


#Logistic Regression 
lr = LogisticRegression()        # instantiate model
lr.fit(X_train, y_train)         # train the machine model


# In[ ]:


#predict
lr_p = lr.predict(X_test)


# In[ ]:


lr_p


# ### Gaussian NB

# In[69]:


# Gaussian Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)


# In[70]:


nb_pred = nb.predict(X_test)


# In[71]:


nb_pred


# ### Random Forest Classifier

# In[72]:


# Random Forest Classifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)


# In[73]:


rf_pred = rf.predict(X_test)


# In[74]:


rf_pred


# ### Support Vector Classifier

# In[75]:


# Support Vector Classifier
svc = SVC()
svc.fit(X_train, y_train)


# In[76]:


svc = svc.predict(X_test)


# In[77]:


svc


# ## 3.3 Model Evaluation
# The metrics used for the model evaluation are Recall_Score, Precision_Score, F1_Score, and Accuracy_Score. Recall_Score is the primary metric as it measures the model's ability to identify correctly customers that will churn. It reduces the probability of classifying a customer who will Churn as No-Churn.

# In[78]:


outcome = classification_report(y_test, rf_pred, labels=[0,1])
print(outcome)


# In[79]:


outcome = classification_report(y_test, lr_p, labels=[0,1])
print(outcome)


# In[80]:


outcome = classification_report(y_test, svc, labels=[0,1])
print(outcome)


# In[ ]:





# 
# - With Recall_Score being the primary metric, most algorithms did not perform well though the other metrics of Precision_Score, F1_Score and Accuracy_Score were better. Naive Bayes algorithm with a rate of 73.44% gave the best Recall_Score with the other algorithms hovering around 50%.

# In[ ]:





# ## Hyperparameters Tuning
# The hyperparameters of the various algorithms are tuned to improve the Recall_Score using GridSearchCV

# In[81]:


# Logistic Regression
param_grid = {'solver': ['newton-cg', 'lbfgs', 'liblinear'],
              'C': [0.001, 0.01, 1, 10, 100],
              'penalty': ['l1', 'l2', 'elasticnet', None]}

grid_search = GridSearchCV(lr, param_grid, scoring='recall', cv=10)

grid_search.fit(X_train, y_train)

print(f'Best Recall_Score: {grid_search.best_score_}, for {grid_search.best_params_}')


# In[82]:


# Naive Bayes
param_grid = {'var_smoothing': np.logspace(0, -9, num=10)}

grid_search = GridSearchCV(nb, param_grid, scoring='recall', cv=10)

grid_search.fit(X_train, y_train)

print(f'Best Recall_Score: {grid_search.best_score_}, for {grid_search.best_params_}')


# In[83]:


# Random Forest
param_grid = {'n_estimators': [50, 100, 150],
              'max_depth': [None, 10, 20, 30]}

grid_search = GridSearchCV(rf, param_grid, scoring='recall', cv=10)

grid_search.fit(X_train, y_train)

print(f'Best Recall_Score: {grid_search.best_score_}, for {grid_search.best_params_}')


# In[84]:


# Support Vector Machines
param_grid = {'C': [0.1, 1, 10],
              'kernel': ['linear', 'rbf'],
              'gamma': ['scale', 'auto', 0.1, 1]}

grid_search = GridSearchCV(svc, param_grid, scoring='recall', cv=10)

grid_search.fit(X_train, y_train)

print(f'Best Recall_Score: {grid_search.best_score_}, for {grid_search.best_params_}')


# In[ ]:





# ##  Building and Evaluating Model using Naive Bayes (tuned hyperparameter)

# In[ ]:


nb_tuned = GaussianNB(var_smoothing= 1.0)
nb_tuned.fit(X_train, y_train)
nb_tuned_pred = nb_tuned.predict(X_test)

print('Classification Report: Naive Bayes Model')
print('----------------------------------------')
print(classification_report(y_test, nb_tuned_pred))

fig, ax = plt.subplots()
sns.heatmap(confusion_matrix(y_test, nb_tuned_pred, normalize='true'), annot=True, cmap='coolwarm')
ax.set_title('Confusion Matrix')
ax.set_ylabel('Actual')
ax.set_xlabel('Predicted');


# In[ ]:





# *Observation*: 
# - The selected algorithm for the Customer Churn Prediction model is Naive Bayes with an improved Recall_Score of 78%. That is out of every 5 churn customers the model can identify about 4 of them correctly.

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




