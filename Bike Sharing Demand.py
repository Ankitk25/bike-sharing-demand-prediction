#!/usr/bin/env python
# coding: utf-8

# ## **Bike Sharing Demand Prediction**

# Project Type - Regression
# 
# Contribution - Team Project
# 
# Submitted by- Durgesh,Ram,Aditya,Uday

# ## **Problem Statement**
# Currently Rental bikes are introduced in many urban cities for the enhancement of mobility comfort. It is important to make the rental bike available and accessible to the public at the right time as it lessens the waiting time. Eventually, providing the city with a stable supply of rental bikes becomes a major concern. The crucial part is the prediction of bike count required at each hour for the stable supply of rental bikes

# In[2]:


# Standard Libraries import for data handling and manipulation of dataset
import numpy as np
import pandas as pd
from numpy import math
#For handling date column
from datetime import datetime

#For visualization purpose
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.graph_objects as px
import plotly.express as x

#For feature selection
from sklearn import feature_selection
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[3]:


day_df=pd.read_csv('/content/drive/MyDrive/day.csv')
hour_df=pd.read_csv('/content/drive/MyDrive/hour.csv')


# Dataset First View

# In[4]:


day_df.head()


# In[5]:


hour_df.head()


# Dataset Rows & Columns count

# In[10]:


# Dataset Rows & Columns count
day_df.shape,hour_df.shape


# Dataset Information

# In[22]:


day_df.info()


# In[23]:


hour_df.info()


# In[20]:


day_df.isnull()


# In[21]:


hour_df.isnull()


# Dataset Describe

# In[18]:


day_df.describe()


# In[19]:


hour_df.describe()


# **Variables Description**
# 
# *  datetime - hourly date + timestamp
# *  season - 1 = spring, 2 = summer, 3 = fall, 4 = winter
# *  holiday - whether the day is considered a holiday
# *  workingday - whether the day is neither a weekend nor holiday
# *  weather - `1: Clear, Few clouds, Partly cloudy, Partly cloudy. 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
# *  temp - temperature in Celsius
# *  atemp - "feels like" temperature in Celsius
# *  humidity - relative humidity
# *  windspeed - wind speed
# *  casual - number of non-registered user rentals initiated
# *  registered - number of registered user rentals initiated
# *  count - number of total rentals
# 
# 
# 
# 

# In[24]:


plt.figure(figsize=(10, 6))
plt.boxplot(hour_df['cnt'])
plt.title('Boxplot of Bike Sharing Count')
plt.ylabel('Count')
plt.show()


# In[25]:


# Calculate the z-scores for each data point in the "count" variable
z_scores = (hour_df['cnt'] - hour_df['cnt'].mean()) / hour_df['cnt'].std()


# In[26]:


# Identify outliers based on a threshold (e.g., z-score > 3)
outliers = hour_df[z_scores > 3]

# Print the outliers
print("Outliers:")
print(outliers)


# In[27]:


# Visualize the outliers
plt.figure(figsize=(10, 6))
plt.scatter(hour_df.index, hour_df['cnt'], label='Data')
plt.scatter(outliers.index, outliers['cnt'], color='red', label='Outliers')
plt.title('Bike Sharing Count')
plt.xlabel('Index')
plt.ylabel('Count')
plt.legend()
plt.show()


# Handling outliers

# In[28]:


#lineplot of datewise bike sharing demand
count_date=hour_df.groupby(['hr'])['cnt'].sum().reset_index()
fig = plt.figure(figsize=(25,6))
ax = plt.axes()
x = count_date['hr']
ax.plot(x, count_date['cnt'])
plt.show()


# In[29]:


#Lineplot of monthly bike sharing demand
count_date=hour_df.groupby(['mnth'])['cnt'].sum().reset_index()
fig = plt.figure(figsize=(25,6))
ax = plt.axes()
x = count_date['mnth']
ax.plot(x, count_date['cnt'])
plt.show()


# ### Data Wrangling

# In[30]:


hour_df.rename(columns={'instant':'rec_id','dteday':'datetime','holiday':'is_holiday','workingday':'is_workingday',
                        'weathersit':'weather_condition','hum':'humidity','mnth':'month',
                        'cnt':'total_count','hr':'hour','yr':'year'},inplace=True)
hour_df.head()


# ##Data Vizualization
# ### Understand the relationships between variables

# Seasonwise Hourly distribution of bikes

# In[32]:


fig,ax=plt.subplots(figsize=(15,6))
sns.set_style('white')

sns.pointplot(x='hour',y='total_count',data=hour_df[['hour','total_count','season']],hue='season',ax=ax)
ax.set_title('Season wise hourly distribution of counts')
plt.show()


# In[34]:


fig,ax1=plt.subplots(figsize=(15,8))
sns.boxplot(x='hour',y='total_count',data=hour_df[['hour','total_count']],ax=ax1)
ax1.set_title('Season wise hourly distribution of counts')
plt.show()


# Weekdaywise hourly distribution of count.

# In[35]:


fig,ax=plt.subplots(figsize=(20,8))
sns.pointplot(x='hour',y='total_count',data=hour_df[['hour','total_count','weekday']],hue='weekday')
ax.set_title('Weekday wise hourly distribution of counts')
plt.show()


# Monthly distribution of counts.

# In[36]:


fig,ax1=plt.subplots(figsize=(20,8))
sns.barplot(x='month',y='total_count',data=hour_df[['month','total_count']],ax=ax1)
ax1.set_title('Monthly distribution of counts')
plt.show()


# In[37]:


fig,ax2=plt.subplots(figsize=(20,8))
sns.barplot(x='month',y='total_count',data=hour_df[['month','total_count','season']],hue='season',ax=ax2)
ax2.set_title('Season wise monthly distribution of counts')
plt.show()


# Yearwise Distribution of counts.

# In[38]:


fig,ax=plt.subplots(figsize=(20,8))
sns.violinplot(x='year',y='total_count',data=hour_df[['year','total_count']])
ax.set_title('Yearly wise distribution of counts')
plt.show()


# In[39]:


fig,(ax1,ax2) = plt.subplots(ncols=2,figsize=(20,5))
sns.barplot(data=hour_df,x='is_holiday',y='total_count',hue='season',ax=ax1)
ax1.set_title('is_holiday wise distribution of counts')
sns.barplot(data=hour_df,x='is_workingday',y='total_count',hue='season',ax=ax2)
ax2.set_title('is_workingday wise distribution of counts')
plt.show()


# Heatmap

# In[40]:


correMtr=hour_df[["temp","atemp","humidity","windspeed","total_count"]].corr()
mask=np.array(correMtr)
mask[np.tril_indices_from(mask)]=False
fig,ax=plt.subplots(figsize=(20,5))
sns.heatmap(correMtr,mask=mask,vmax=0.8,square=True,annot=True,ax=ax)
ax.set_title('Correlation matrix of attributes')
plt.show()


# Impact of Holidays

# In[41]:


#Barplot of holiday vs rented bike count
y=hour_df.groupby('is_holiday')['total_count'].mean().reset_index()
fig = plt.subplots(figsize=(6, 4))
sns.barplot(x ='is_holiday',
            y ='total_count',
            data = y).set_title('Average Bike demand during Holidays')
plt.show()


# ##Model Training

# In[43]:


#dropping id column as it is irrelevant to our analysis
hour_df=hour_df.drop(labels=['rec_id','datetime','casual','registered'],axis=1)


# In[44]:


# Seprating Independent and dependent features
X = hour_df.drop(labels=['total_count'],axis=1)
Y = hour_df[['total_count']]


# In[45]:


#checking target variable
Y


# In[46]:


# Defining which columns should be ordinal-encoded and which should be scaled
categorical_cols = X.select_dtypes(include='object').columns
numerical_cols = X.select_dtypes(exclude='object').columns


# In[47]:


numerical_cols


# In[48]:


# for handling Missing Values
from sklearn.impute import SimpleImputer

# for Feature Scaling
from sklearn.preprocessing import StandardScaler

# for Ordinal Encoding
from sklearn.preprocessing import OrdinalEncoder

## for creating pipelines
from sklearn.pipeline import Pipeline

#for combining pipelines
from sklearn.compose import ColumnTransformer


# In[49]:


# Numerical Pipeline
num_pipeline=Pipeline(
    steps=[
    ('imputer',SimpleImputer(strategy='median')),
    ('scaler',StandardScaler())

    ]

)

# Categorigal Pipeline
cat_pipeline=Pipeline(
    steps=[
    ('imputer',SimpleImputer(strategy='most_frequent')),
    ('ordinalencoder',OrdinalEncoder(categories=[])),
    ('scaler',StandardScaler())
    ]

)

preprocessor=ColumnTransformer([
('num_pipeline',num_pipeline,numerical_cols),
('cat_pipeline',cat_pipeline,categorical_cols)
])


# In[50]:


## Train test split
from sklearn import preprocessing

# Example usage of a preprocessing module
scaler = preprocessing.StandardScaler()

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.30,random_state=30)


# In[51]:


#scaling dataset
X_train=pd.DataFrame(preprocessor.fit_transform(X_train),columns=preprocessor.get_feature_names_out())
X_test=pd.DataFrame(preprocessor.transform(X_test),columns=preprocessor.get_feature_names_out())


# In[52]:


#checking for scaling
X_train.head()


# In[53]:


## Model Training

from sklearn.linear_model import LinearRegression,Lasso,Ridge,ElasticNet
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor

#for metric evaluation
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error


# In[54]:


#function to evaluate model using mae,rmse and R2 score
import numpy as np
def evaluate_model(true, predicted):
    mae = mean_absolute_error(true, predicted)
    mse = mean_squared_error(true, predicted)
    rmse = np.sqrt(mean_squared_error(true, predicted))
    r2_square = r2_score(true, predicted)
    return mae, rmse, r2_square


# In[55]:


## Training  multiple models

models={
    'LinearRegression':LinearRegression(),
    'Random Forest':RandomForestRegressor(),
    'Extra Trees Regressor':ExtraTreesRegressor()
}
trained_model_list=[]
model_list=[]
r2_list=[]

for i in range(len(list(models))):
    model=list(models.values())[i]
    model.fit(X_train,y_train)

    #Make Predictions
    y_pred=model.predict(X_test)

    mae, rmse, r2_square=evaluate_model(y_test,y_pred)

    print(list(models.keys())[i])
    model_list.append(list(models.keys())[i])

    print('Model Training Performance')
    print("RMSE:",rmse)
    print("MAE:",mae)
    print("R2 score",r2_square*100)

    r2_list.append(r2_square)

    print('='*35)
    print('\n')


# In[ ]:




