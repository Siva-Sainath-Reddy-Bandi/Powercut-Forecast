#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import numpy as np


# In[192]:


data = pd.read_csv("C:/Users/Code/OneDrive/Desktop/outage_data.csv")
df = pd.DataFrame(data)


# In[194]:


data.head()


# In[195]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 


# In[263]:


train = data.drop(['Year','Event Description','Geographic Areas','NERC Region','Respondent','Date Event Began','Date of Restoration','Tags', 'Time Event Began','Time of Restoration', 'Number of Customers Affected'], axis=1)
test = data[['Number of Customers Affected', 'Time Event Began', 'Time of Restoration']]


# In[264]:


X_train, X_test, y_train, y_test = train_test_split(train, test, test_size=0.7, random_state=2)


# In[265]:


regr = LinearRegression()


# In[266]:


regr.fit(X_train, y_train)


# In[267]:


pred = regr.predict(X_test)


# In[268]:


pred


# In[269]:


regr.score(X_test, y_test)

