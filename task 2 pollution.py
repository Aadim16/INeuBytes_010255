#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plot;
import seaborn as sns
from statsmodels.tsa.stattools import acf, pacf
from math import sqrt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_log_error 
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score


# In[31]:


pollution_data = pd.read_csv('PRSA_data_2010.1.1-2014.12.31.csv')


# In[32]:


pollution_data.head()


# In[33]:


pollution_data.tail()


# In[34]:


# to get a statically summary from each column of dataset
pollution_data.describe()


# In[35]:


pollution_data.info()


# In[36]:


pollution_data.shape


# In[37]:


pollution_data.isna().any()


# In[38]:


data = pollution_data['pm2.5'].isna().sum()
print(f'total number of na in pm2.5 : {data}')


# In[39]:


mean_of_pollution_data = pollution_data['pm2.5'].mean()
print(f'mean of pollution_data = {mean_of_pollution_data}')


# In[40]:


median_of_pollution_data = pollution_data['pm2.5'].median()
print(f'median of pollution_data = {median_of_pollution_data}')


# In[41]:


pollution_data.describe()


# In[44]:


pollution_data.dropna(inplace=True)


# In[14]:


X = pollution_data[['TEMP', 'PRES', 'DEWP', 'Iws', 'Is' , 'Ir']]
y = pollution_data['pm2.5']


# In[15]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)


# In[16]:


# scaler ==std_scale
std_scale = StandardScaler()
X_train_scaled = std_scale.fit_transform(X_train)
X_test_scaled = std_scale.transform(X_test)


# In[17]:


linear_regression_model = LinearRegression()


# In[18]:


linear_regression_model.fit(X_train_scaled, y_train)


# In[19]:


y_pred = linear_regression_model.predict(X_test_scaled)


# In[20]:


mean_sq_error = mean_squared_error(y_test, y_pred)
mean_abs_error = mean_absolute_error (y_test, y_pred)
print(f'mean square error : {mean_sq_error}')
print(f'mean absolute error : {mean_abs_error}')


# In[21]:


pseudo_data = np.array([[12.448521 , 1016.447654 , 1.817246 , 23.889140,0.052734,0.194916]])


# In[22]:


new_data_scaled = std_scale.transform(pseudo_data);


# In[23]:


prediction = linear_regression_model.predict(new_data_scaled)


# In[24]:


print(f'Predicted value for pm2.5: {prediction[0]}')


# In[25]:


np.random.seed(0)
X = np.random.rand(100, 5)
y = np.random.rand(100)


# In[26]:


cv_scores = cross_val_score(linear_regression_model, X_train, y_train, cv=10)


# In[27]:


print("Cross-Validation Scores ", cv_scores)
print("Mean of cross validation scores:", np.mean(cv_scores))


# In[28]:


train_score = linear_regression_model.score(X_train, y_train)
test_score = linear_regression_model.score(X_test, y_test)
print("Training R^2 Score:", train_score)
print("Test R^2 Score:", test_score)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




