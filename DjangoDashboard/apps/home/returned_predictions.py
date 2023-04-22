#!/usr/bin/env python
# coding: utf-8

# # Libraries

# In[353]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import tqdm
import hyperopt
import sys
import scipy

import lightgbm
#from catboost import Pool, CatBoostClassifier, cv, CatboostIpythonWidget

from datetime import datetime
from sklearn.preprocessing import LabelEncoder
# sklearn.cross_validation import cross_val_score, KFold
from sklearn.metrics import log_loss
from sklearn.model_selection  import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import accuracy_score
from IPython.display import display, HTML








#Importer les librairies necessaires:


from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")

# Libraries for data visualization
import seaborn as sns
import plotly.express as px
from tqdm.notebook import tqdm
import pygrametl
from pygrametl.tables import Dimension
from pygrametl.tables import FactTable


#!pip install -U imbalanced-learn


# In[354]:


import pyodbc 
conn2 = pyodbc.connect('Driver={SQL Server Native Client 11.0};'
                      'Server=sql8004.site4now.net;'
                      'Database=db_a92253_innovision;'
                      'UID=db_a92253_innovision_admin;'
                      'PWD=innovision2022;'
                      )
cursor2 = conn2.cursor()


# In[355]:


SQL_Query = pd.read_sql_query('''select * FROM db_a92253_innovision.dbo.factSales''', conn2)
data = pd.DataFrame(SQL_Query)


# # input data

# In[ ]:





# In[ ]:





# In[356]:


X = data.drop("Returned", axis=1)
y = data["Returned"]


# In[357]:


from sklearn.model_selection import train_test_split
  
# split into 70:30 ration
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 0)
  


# In[358]:



from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state = 2)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())
  


# In[ ]:





# In[ ]:





# In[ ]:





# In[359]:


X_train_res['Returned'] = y_train_res.tolist()


# In[360]:


data= X_train_res


# # Dividing variables into categorical and Numeric

# In[361]:


data = data.drop(['FK_Customer','FK_Ship','FK_Date_Order','FK_Date_Ship','FK_Order','FK_Localisation',],axis=1)
vars=data.dtypes
categorical=[]
numeric=[]
for i in range(0,len(vars)):
    if vars[i]=="object": 
        categorical.append(data.columns[i])
    else:
        numeric.append(data.columns[i]) 


# In[ ]:





# In[ ]:





# In[ ]:





# # Variable processing

# In[362]:


#lable encoding for categorical variables
df1=data[categorical].apply(LabelEncoder().fit_transform)
df2=data[numeric]
df3=pd.concat([df1, df2], axis=1)


# In[ ]:





# # Split data in Train and Test datsets

# In[363]:


train, test = train_test_split(df3, test_size=0.2)
Returned_X =train['Returned'] 
train = train.drop(['Returned'],axis=1)
Returned_Y =test['Returned'] 
test = test.drop(['Returned'],axis=1)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# # Xgboost Model

# In[364]:


gbm = xgb.XGBClassifier(max_depth=2, n_estimators=100, learning_rate=0.8).fit(train,Returned_X)
predictions = gbm.predict(test)


# In[ ]:





# In[365]:


test['returned_predictions']=predictions
test['returned_actual']=Returned_Y


# In[ ]:





# In[366]:


sum_returned=test.groupby(['FK_Product']).sum()


# In[367]:


sum_returned
index_returned = sum_returned[ (sum_returned['returned_predictions'] != sum_returned['returned_actual'])] .index
sum_returned.drop(index_returned , inplace=True)
index_returned = sum_returned[ (sum_returned['returned_predictions'] == 0) & (sum_returned['returned_actual'] == 0) ].index
sum_returned.drop(index_returned , inplace=True)
sum_returned = sum_returned.drop(['Quantity','Profit','Sales','Discount','Ship_Coast','Ship_Duration','returned_actual'],axis=1)


# In[368]:


sum_returned=sum_returned.sort_values(by=['returned_predictions'],ascending=False)


# In[369]:


SQL_Query = pd.read_sql_query('''select * FROM db_a92253_innovision.dbo.DimProduct''', conn2)
DimProduct = pd.DataFrame(SQL_Query)


# In[370]:


DimProduct=DimProduct.rename(columns={"Product_PK": "FK_Product"})


# In[ ]:





# In[371]:


dataCopy=sum_returned
dataCopy=pd.merge(dataCopy, DimProduct, left_on='FK_Product', right_on='FK_Product')
dataCopy= dataCopy.drop(['FK_Product','Product ID','Unit_Price'],axis=1)         


# In[372]:


df_top10_returned=dataCopy[:10]
df_top10_returned=df_top10_returned.rename(columns={"Product Name": "Product_Name"})

# In[373]:


df_top10_returned


# In[374]:


df_category_returned=dataCopy
df_category_returned= dataCopy.drop(['Sub-Category','Product Name'],axis=1) 
df_category_returned=df_category_returned.groupby(['Category'],as_index=False).sum()
df_category_returned=df_category_returned.sort_values(by=['returned_predictions'],ascending=False)


# In[375]:


df_sub_category_returned=dataCopy
df_sub_category_returned= dataCopy.drop(['Category','Product Name'],axis=1) 
df_sub_category_returned=df_sub_category_returned.rename(columns={"Sub-Category": "Sub_Category"})
df_sub_category_returned=df_sub_category_returned.groupby(['Sub_Category'],as_index=False).sum()
df_sub_category_returned=df_sub_category_returned.sort_values(by=['returned_predictions'],ascending=False)
df_sub_category_returned=df_sub_category_returned[:10]
df_sub_category_returned


# In[ ]:




