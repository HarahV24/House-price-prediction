#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
get_ipython().run_line_magic('matplotlib', 'inline')


# In[6]:


train = pd.read_csv("C:\\Users\\harsh\\Downloads\\train.csv")


# In[7]:


train.head()


# In[8]:


train.shape


# In[9]:


train.info()


# In[10]:


plt.subplots(figsize=(12,9))
sns.distplot(train['SalePrice'], fit=stats.norm)
# Get the fitted parameters used by the function
(mu, sigma) = stats.norm.fit(train['SalePrice'])
# plot with the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
#Probablity plot
fig = plt.figure()
stats.probplot(train['SalePrice'], plot=plt)
plt.show()


# In[11]:


#we use log function which is in numpy
train['SalePrice'] = np.log1p(train['SalePrice'])
#Check again for more normal distribution
plt.subplots(figsize=(12,9))
sns.distplot(train['SalePrice'], fit=stats.norm)
# Get the fitted parameters used by the function
(mu, sigma) = stats.norm.fit(train['SalePrice'])
# plot with the distribution
plt.legend(['Normal dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc='best')
plt.ylabel('Frequency')
#Probablity plot
fig = plt.figure()
stats.probplot(train['SalePrice'], plot=plt)
plt.show()


# In[12]:


#Let's check if the data set has any missing values. 
train.columns[train.isnull().any()]


# In[13]:


#plot of missing value attributes
plt.figure(figsize=(12, 6))
sns.heatmap(train.isnull())
plt.show()


# In[14]:


#missing value counts in each of these columns
Isnull = train.isnull().sum()/len(train)*100
Isnull = Isnull[Isnull>0]
Isnull.sort_values(inplace=True, ascending=False)
Isnull


# In[15]:


#Convert into dataframe
Isnull = Isnull.to_frame()
Isnull.columns = ['count']
Isnull.index.names = ['Name']
Isnull['Name'] = Isnull.index


# In[16]:


#plot Missing values
plt.figure(figsize=(13, 5))
sns.set(style='whitegrid')
sns.barplot(x='Name', y='count', data=Isnull)
plt.xticks(rotation = 90)
plt.show()


# In[17]:


train_corr = train.select_dtypes(include=[np.number])


# In[18]:


train_corr.shape


# In[19]:


del train_corr['Id']


# In[20]:


#Coralation plot
corr = train_corr.corr()
plt.subplots(figsize=(20,9))
sns.heatmap(corr, annot=True)


# In[21]:


top_feature = corr.index[abs(corr['SalePrice']>0.5)]
plt.subplots(figsize=(12, 8))
top_corr = train[top_feature].corr()
sns.heatmap(top_corr, annot=True)
plt.show()


# In[22]:


#unique value of OverallQual
train.OverallQual.unique()


# In[27]:


sns.barplot()
(train.OverallQual, train.SalePrice)


# In[24]:


#boxplot
plt.figure(figsize=(18, 8))
sns.boxplot(x=train.OverallQual, y=train.SalePrice)


# In[30]:


col = ['SalePrice', 'OverallQual', 'GrLivArea', 'GarageCars']
sns.set(style='ticks')
sns.pairplot(train[col], size=4, kind='reg')


# In[31]:


print("Find most important features relative to target")
corr = train.corr()
corr.sort_values(['SalePrice'], ascending=False, inplace=True)
corr.SalePrice


# In[34]:


train['PoolQC'] = train['PoolQC'].fillna('None')
#Arround 50% missing values attributes have been fill by None
train['MiscFeature'] = train['MiscFeature'].fillna('None')
train['Alley'] = train['Alley'].fillna('None')
train['Fence'] = train['Fence'].fillna('None')
train['FireplaceQu'] = train['FireplaceQu'].fillna('None')
#Group by neighborhood and fill in missing value by the median LotFrontage of all the neighborhood
train['LotFrontage'] = train.groupby("Neighborhood")["LotFrontage"].transform(
    lambda x: x.fillna(x.median()))
#GarageType, GarageFinish, GarageQual and GarageCond these are replacing with None
for col in ['GarageType', 'GarageFinish', 'GarageQual', 'GarageCond']:
    train[col] = train[col].fillna('None')


# In[35]:


for col in ['GarageYrBlt', 'GarageArea', 'GarageCars']:
    train[col] = train[col].fillna(int(0))
#BsmtFinType2, BsmtExposure, BsmtFinType1, BsmtCond, BsmtQual these are replacing with None
for col in ('BsmtFinType2', 'BsmtExposure', 'BsmtFinType1', 'BsmtCond', 'BsmtQual'):
    train[col] = train[col].fillna('None')
#MasVnrArea : replace with zero
train['MasVnrArea'] = train['MasVnrArea'].fillna(int(0))
#MasVnrType : replace with None
train['MasVnrType'] = train['MasVnrType'].fillna('None')


# In[36]:


train['Electrical'] = train['Electrical'].fillna(train['Electrical']).mode()[0]
#There is no need of Utilities
train = train.drop(['Utilities'], axis=1)
#Checking there is any null value or not
plt.figure(figsize=(10, 5))
sns.heatmap(train.isnull())


# In[37]:


cols = ('FireplaceQu', 'BsmtQual', 'BsmtCond', 'GarageQual', 'GarageCond', 
        'ExterQual', 'ExterCond','HeatingQC', 'PoolQC', 'KitchenQual', 'BsmtFinType1', 
        'BsmtFinType2', 'Functional', 'Fence', 'BsmtExposure', 'GarageFinish', 'LandSlope',
        'LotShape', 'PavedDrive', 'Street', 'Alley', 'CentralAir', 'MSSubClass', 'OverallCond', 
        'YrSold', 'MoSold', 'MSZoning', 'LandContour', 'LotConfig', 'Neighborhood',
        'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle', 'RoofMatl', 'Exterior1st',
        'Exterior2nd', 'MasVnrType', 'MasVnrArea', 'Foundation', 'GarageType', 'MiscFeature', 
        'SaleType', 'SaleCondition', 'Electrical', 'Heating')
from sklearn.preprocessing import LabelEncoder
for c in cols:
    lbl = LabelEncoder() 
    lbl.fit(list(train[c].values)) 
    train[c] = lbl.transform(list(train[c].values))


# In[38]:


#Take the targate variable into y
y = train['SalePrice']
#Delete the saleprice
del train['SalePrice']
#Take their values in X and y
X = train.values
y = y.values
# Split data into train and test formate
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=7)


# In[41]:


#Train the model
from sklearn import linear_model
model = linear_model.LinearRegression()


# In[43]:


#Prediction
print("Predict value " + str(model.predict([X_test[142]])))
print("Real value " + str(y_test[142]))


# In[44]:


#Score/Accuracy
print("Accuracy --> ", model.score(X_test, y_test)*100)


# In[45]:


#Train the model
from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor(n_estimators=1000)
#Fit
model.fit(X_train, y_train)


# In[46]:


print("Accuracy --> ", model.score(X_test, y_test)*100)


# In[47]:


#Train the model
from sklearn.ensemble import GradientBoostingRegressor
GBR = GradientBoostingRegressor(n_estimators=100, max_depth=4)
#Fit
GBR.fit(X_train, y_train)


# In[48]:


print("Accuracy --> ", GBR.score(X_test, y_test)*100)


# In[ ]:


Now the project is runnig.
and we use this project.
Made By Harsh Vaish

