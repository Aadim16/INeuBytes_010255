#!/usr/bin/env python
# coding: utf-8

# import pandas as pd
# import matplotlib.pyplot as plt;
# import seaborn as sns
# from statsmodels.tsa.stattools import acf, pacf
# from math import sqrt
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# from sklearn.metrics import mean_absolute_error 
# from sklearn.preprocessing import StandardScaler
# from sklearn.metrics import mean_squared_log_error 
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import cross_val_score
# from sklearn.tree import DecisionTreeRegressor
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.metrics import mean_squared_error, r2_score

# In[2]:


import pandas as pd
import matplotlib.pyplot as plt;
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
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# In[3]:


try:
    data = pd.read_csv('data.csv', encoding='ISO-8859-1')
except Exception as e:
    print(f"Error reading the CSV file: {e}")


# In[4]:


data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])


# In[5]:


data.head()


# In[6]:


data.tail()


# In[7]:


# to get statically summary use describe function
data.describe()


# In[8]:


data.info()


# In[9]:


data.shape


# In[10]:


data.isna().any()


# In[11]:


data.isna().sum()


# In[12]:


# removing all the null values since we cannot replace it with any statical value
data.dropna(inplace=True)


# In[13]:


data.isna().any()


# In[14]:


quantity = data['Quantity']
uniprice = data['UnitPrice']
plt.figure(figsize=(10, 6))
plt.hist(quantity, bins=30, alpha=0.5, label='quantity')
plt.hist(uniprice, bins=30, alpha=0.5, label='uniprice')
plt.xlabel('quantity')
plt.ylabel('price of each single product')
plt.title('Histogram between quantity and uniprice')
plt.legend()
plt.show()
# plt.hexbin(quantity, uniprice, gridsize=30, cmap='Blues')


# In[15]:


plt.scatter(quantity, uniprice, marker='o', color='blue', alpha=0.5)
plt.grid(True)
plt.show()


# In[16]:


quantity = data['Quantity']
customerid = data['CustomerID']
plt.figure(figsize=(10, 6))
plt.hist(customerid, bins=30, alpha=0.5, label='CustomerID')
plt.hist(quantity, bins=30, alpha=0.5, label='quantity')
plt.xlabel('quantity')
plt.ylabel('quantity of product bought by each')
plt.title('Histogram between quantity and coustmerid')
plt.legend()
plt.show()
# plt.hexbin(quantity, customerid, gridsize=30, cmap='Blues')


# In[17]:


plt.scatter(quantity, customerid, marker='o', color='blue', alpha=0.5)
plt.grid(True)
plt.show()


# In[18]:


data['Total_Spent_amount'] = data['Quantity'] * data['UnitPrice']
data['InvoiceDate'] = pd.to_datetime(data['InvoiceDate'])
data['Recency'] = (pd.to_datetime('today') -data['InvoiceDate']).dt.days
data = data.groupby('CustomerID').agg({
    'Total_Spent_amount': 'sum',
    'InvoiceNo': 'nunique',
    'InvoiceDate': 'max'
}).reset_index()


# In[19]:


data['Total_Spent_amount'].head()


# In[20]:


data['Total_Spent_amount'].tail()


# In[21]:


yearly_amount_spent = data.groupby('CustomerID')['Total_Spent_amount'].sum().reset_index()
yearly_amount_spent.columns = ['CustomerID', 'yearly_amount_spent']


# In[22]:


X_value = yearly_amount_spent[['CustomerID']]
y_value = yearly_amount_spent['yearly_amount_spent']


# In[23]:


X_train, X_test, y_train, y_test = train_test_split(X_value, y_value, test_size=0.2, random_state=42)


# In[31]:


customer_data = data.groupby('CustomerID').agg({
    'Total_Spent_amount': 'sum',
#     'Quantity': 'sum',
    'InvoiceNo': 'nunique'
}).reset_index()


# In[27]:


linear_regression = LinearRegression()
decission_regression = DecisionTreeRegressor(random_state=42)
random_forest_regression = RandomForestRegressor(random_state=42)
models = {'Linear Regression': linear_regression, 'Decision Tree': decission_regression , 'Random Forest': random_forest_regression}


# In[30]:


scaler = StandardScaler()
data_scaled = scaler.fit_transform(data[['Total_Spent_amount','InvoiceNo']])

kmeans = KMeans(n_clusters=2, random_state=42)
data['Cluster'] = kmeans.fit_predict(data_scaled)

# Evaluate clustering
silhouette_avg = silhouette_score(data_scaled,data['Cluster'])
print(f'Silhouette Score: {silhouette_avg}')


# In[33]:


import numpy as np
X = np.random.rand(100, 5)
y = np.random.rand(100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)
lr_mse = mean_squared_error(y_test, lr_predictions)
lr_r2 = r2_score(y_test, lr_predictions)

print("Linear Regression Performance:")
print("Mean Squared Error:", lr_mse)
print("R^2 Score:", lr_r2)

# Decision Tree
dt_model = DecisionTreeRegressor(random_state=42)
dt_model.fit(X_train, y_train)
dt_predictions = dt_model.predict(X_test)
dt_mse = mean_squared_error(y_test, dt_predictions)
dt_r2 = r2_score(y_test, dt_predictions)

print("\nDecision Tree Performance:")
print("Mean Squared Error:", dt_mse)
print("R^2 Score:", dt_r2)

# Random Forest
rf_model = RandomForestRegressor(random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

print("\nRandom Forest Performance:")
print("Mean Squared Error:", rf_mse)
print("R^2 Score:", rf_r2)


# In[ ]:





# In[ ]:





# In[ ]:




