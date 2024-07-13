#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd;
import numpy as np;
import matplotlib.pyplot as plot;
from matplotlib import rcParams;
from matplotlib.cm import rainbow; 
import seaborn as sns;
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split as tts
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression


# In[3]:


# reading our file heart.csv using pandas
heart_data=pd.read_csv('heart.csv')


# In[4]:


# getting shape of our dataset
heart_data.shape


# In[5]:


# printing first 5 rows
heart_data.head()


# In[6]:


# printing last 5 rows
heart_data.tail()


# In[7]:


# to get a statically summary from each column of dataset
heart_data.describe()


# In[8]:


heart_data.info()


# In[9]:


# ploting the number of plositive and negitive tests
# heart_data['target'].value_counts().plot(kind='bar');
sns.countplot(heart_data['target'])
plot.title("total number of positive and negitive cases")
plot.xlabel("result 0/negitive_test and 1/positive_test")
plot.ylabel("number of zeroes and ones")
plot.show();


# In[10]:


# getting the data summry in form of heatmap
cor=heart_data.corr()
top_corr_feature=cor.index
plot.figure(figsize=(20,20))
sns.heatmap(heart_data[top_corr_feature].corr(),annot=True,cmap="RdYlGn")
plot.title("HEATMAP")
plot.show()


# In[11]:


# to train our model we are making a dummy copy of our dataset
temp_data = pd.get_dummies(heart_data,columns=['sex','cp','fbs','restecg','exang','slope','ca','thal'])


# In[12]:


temp_data


# In[13]:


sc=StandardScaler()
to_scale=['chol','thalach','oldpeak','age','trestbps']
temp_data[to_scale]=sc.fit_transform(temp_data[to_scale])


# In[14]:


temp_data.head()
temp_data.tail()


# In[33]:


y=temp_data['target']
x=temp_data.drop(['target'],axis=1)


# In[35]:


xtrain,xtest,ytrain,ytest=tts(x,y,test_size=0.2,random_state=42)


# In[ ]:


training_model=MLPClassifier(hidden_layer_sizes=(128,128),max_iter=300)
training_model.fit(xtrain,ytrain)


# In[ ]:


test_prep=training_model.predict(xtest)


# In[ ]:


acc=accuracy_score(ytest,test_prep)
print(acc*100)


# In[23]:


model_name = LinearRegression()


# In[24]:


# spliting data into 10 folds,
# dhuffle randomly shuffle the data besore deviding
# 
Spliting = KFold(n_splits=10, shuffle=True, random_state=42)


# In[25]:


splited_values = []


# In[37]:


for train_index, test_index in Spliting.split(x):
    xtrain, xtest = x.iloc[train_index], x.iloc[test_index]  
    ytrain, ytest = y.iloc[train_index], y.iloc[test_index]
    print(train_index)
    print(test_index)


# In[27]:


model_name.fit(xtrain, ytrain)


# In[28]:


values = model_name.score(xtest, ytest)
splited_values.append(values)


# In[29]:


for i, score in enumerate(splited_values):
    print(f"splitted {i+1} Score: {values}")


# In[30]:


finalmean_value = np.mean(splited_values)


# In[31]:


print(f"mean of cross validation is {finalmean_value}")


# In[32]:


print((finalmean_value)*(100))


# In[ ]:





# In[ ]:





# In[ ]:




