#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
from sklearn.cluster import KMeans



#Importer les librairies necessaires:
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore")


# Libraries for data visualization



# In[6]:


import pyodbc 
conn2 = pyodbc.connect('Driver={SQL Server Native Client 11.0};'
                      'Server=sql8004.site4now.net;'
                      'Database=db_a92253_innovision;'
                      'UID=db_a92253_innovision_admin;'
                      'PWD=innovision2022;'
                      )
cursor2 = conn2.cursor()


# In[23]:


SQL_Query = pd.read_sql_query('''select * FROM db_a92253_innovision.dbo.factSales''', conn2)
data = pd.DataFrame(SQL_Query)


# In[ ]:





# In[56]:


data


# In[57]:


X = data[['Quantity', 'Profit', 'Sales', 'Discount', 'Ship_Coast', 'Ship_Duration']]


# In[58]:


# Appliquer l'algorithme K-Means avec 3 clusters
kmeans = KMeans(n_clusters=2)
kmeans.fit(X)


# In[59]:


data['Cluster'] = kmeans.predict(X)


# In[60]:


data['Cluster'].value_counts()


# In[100]:



customer_profits1 = data.groupby('FK_Customer',as_index=False)['Profit'].sum()

# Trier les clients par profit décroissant et sélectionner les 10 premiers
top_customers = customer_profits1.sort_values(ascending=False,by='Profit').head(5)

# Afficher le top 10 des meilleurs clients
print(top_customers)
top_customers_df = pd.DataFrame(top_customers)
# Afficher le top 10 des meilleurs clients
top_customers_df


# In[101]:


SQL_Query = pd.read_sql_query('''select * FROM db_a92253_innovision.dbo.DimCustomer''', conn2)
DimCustomer = pd.DataFrame(SQL_Query)


# In[102]:


DimCustomer=DimCustomer.rename(columns={"Customer_PK": "FK_Customer"})


# In[103]:


Top5Customers=top_customers_df
Top5Customers=pd.merge(Top5Customers, DimCustomer, left_on='FK_Customer', right_on='FK_Customer')
Top5Customers= Top5Customers.drop(['FK_Customer','Segment'],axis=1) 
Top5Customers


# In[104]:



customer_profits = data.groupby('FK_Localisation',as_index=False)['Profit'].sum()

# Trier les clients par profit décroissant et sélectionner les 10 premiers
top_localisation = customer_profits.sort_values(ascending=False,by='Profit').head(5)

# Afficher le top 10 des meilleurs clients
top_localisation_df = pd.DataFrame(top_localisation)
# Afficher le top 10 des meilleurs clients


# In[105]:


SQL_Query = pd.read_sql_query('''select * FROM db_a92253_innovision.dbo.DimLocalisation''', conn2)
DimLocalisation = pd.DataFrame(SQL_Query)


# In[106]:


DimLocalisation=DimLocalisation.rename(columns={"Localisation_PK": "FK_Localisation"})


# In[107]:


Top5localisation=top_localisation_df
Top5localisation=pd.merge(Top5localisation, DimLocalisation, left_on='FK_Localisation', right_on='FK_Localisation')
Top5localisation= Top5localisation.drop(['FK_Localisation','Country','State','Region','Postal_code','Maket'],axis=1) 
Top5localisation


# In[ ]:




