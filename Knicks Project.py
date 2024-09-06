#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install pandas numpy scikit-learn matplotlib seaborn


# In[2]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import math


# In[3]:


pip install matplotlib


# In[4]:


pip install xlrd


# In[5]:


pip install openpyxl


# In[6]:


game_logs = pd.read_csv(r"C:\Users\steph\Downloads\NBA-BoxScores-2023-2024.csv\NBA-BoxScores-2023-2024.csv",index_col = 1)


# In[7]:


game_logs


# In[8]:


df = pd.DataFrame(game_logs)
df


# In[9]:


nyk_data = df[df['TEAM_ABBREVIATION'] == 'NYK']
nyk_data


# In[ ]:





# In[10]:


nyk_team_totals = nyk_data[['OREB', 'PTS', 'AST', 'PLUS_MINUS' ]]
nyk_team_totals


# In[11]:


nyk_team_totals.dropna()


# In[12]:


nyk_team_totals = nyk_team_totals.rename_axis('Knicks Games')
nyk_team_totals


# In[13]:


nyk_stats_per = nyk_team_totals.groupby('Knicks Games').sum()
nyk_stats_per


# In[14]:


#this data is right ^^^^ its just out of order


# In[24]:


correlation_matrix = nyk_stats_per[['OREB','PTS','AST','PLUS_MINUS']].corr()


# In[26]:


correlation_matrix #0 means no correlation, -1 means perfect negative correlation and vice versa


# In[27]:


import statsmodels.api as sm


# In[28]:


x = nyk_stats_per[['OREB']]
y = nyk_stats_per[['PTS']]
x = sm.add_constant(x)


# In[29]:


model = sm.OLS(y, x).fit()


# In[30]:


print(model.summary())


# In[ ]:




