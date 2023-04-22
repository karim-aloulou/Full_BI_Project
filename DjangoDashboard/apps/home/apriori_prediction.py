#!/usr/bin/env python
# coding: utf-8

# 
# <center>
#     <img src="logo.png" height="300" width="500"/>
#         <h1 style='color:#142b5a; font-weight: bold; font-family: Calibri;font-size: 40px; '>  Prediction of together-bought products with Apriori  </h1>
# </center>
# 
# 
#  <p style='color:#142b5a; font-weight: bold; font-family: Calibri; font-size:25px;'>🎯 ** Objectives **</p>
#  <br>
# <div style='color:#142b5a; font-weight: light; font-family: Calibri;'>In this Notebook we will predict the profit with diffrent methods.</div>
# 
# <p style='color:#142b5a; font-weight: bold; font-family: Calibri; font-size:25px;'>✔️ ** Outline **</p> 
# <br>
# <a href="#import" style='color:#142b5a; font-weight: light; font-family: Calibri;'> • Importing necessary libraries</a>
# <br>
# <br>
# <a href="#clean" style='color:#142b5a; font-weight: light; font-family: Calibri;'> • Importing the dataset</a>
# <br>
# <br>
# <a href="#discover" style='color:#142b5a; font-weight: light; font-family: Calibri;'> • Exploring the dataset</a>
# <br>
# <br>
# <a href="#rename" style='color:#142b5a; font-weight: light; font-family: Calibri;'> • Renaming columns </a>
# <br>
# <a href="#cuf" style='color:#142b5a; font-weight: light; font-family: Calibri;'>• Chosing usefull features</a>
# <br> 
# <a href="#splitting" style='color:#142b5a; font-weight: light; font-family: Calibri; padding:0 0 0 20px'>- Spliting for a better using</a>
# <br>
# <a href="#listprod" style='color:#142b5a; font-weight: light; font-family: Calibri; padding:0 0 0 20px'>- List of bought-together products</a>
# <br>
# <a href="#Apriori" style='color:#142b5a; font-weight: light; font-family: Calibri; padding:0 0 0 20px'>- Application of Apriori</a>
# <br>
# <a href="#Result" style='color:#142b5a; font-weight: light; font-family: Calibri;padding:0 0 0 20px'> - Apriori Results</a>
#         
#   
# </div>
# <br>
# 

# <div style='font-size:100%;'>
#     <a id='import'></a>
#     <h1 style='color: black; font-weight: bold; font-family: Calibri;'>
#         <center> Importing necessary libraries 📚 </center>
#     </h1>
#  
# </div>

# In[34]:


#Installer apyori




# In[35]:


#Importer les Libraries

import pandas as pd 

import pyodbc
          
import os

from tqdm.notebook import tqdm
import pygrametl

from apyori import apriori




# <div style='font-size:100%;'>
#     <a id='clean'></a>
#     <h1 style='color: black; font-weight: bold; font-family: Calibri;'>
#         <center> Importing the dataset ⬇️ </center>
#     </h1>
# </div>

# In[36]:


import pyodbc 




import pyodbc 
conn2 = pyodbc.connect('Driver={SQL Server Native Client 11.0};'
                      'Server=sql8004.site4now.net;'
                      'Database=db_a92253_innovision;'
                      'UID=db_a92253_innovision_admin;'
                      'PWD=innovision2022;'
                      )


cursor2 = pygrametl.ConnectionWrapper(connection=conn2)

# Suppress all warnings
import warnings
warnings.filterwarnings("ignore")

#Reading SQL query into a DataFrame 

#Fact Sales
SQL_Query = pd.read_sql_query('''select * FROM db_a92253_innovision.dbo.FactSales''', conn2)

data = pd.DataFrame(SQL_Query)


# <div style='font-size:100%;'>
#     <a id='discover'></a>
#     <h1 style='color: black; font-weight: bold; font-family: Calibri;'>
#         <center> Exploring the dataset 🧐 </center>
#     </h1>
# </div>

# In[ ]:





# In[37]:


SQL_Query = pd.read_sql_query('''select Product_PK,[Product Name] FROM db_a92253_innovision.dbo.DimProduct''', conn2)
dimProduct = pd.DataFrame(SQL_Query)


# #=== <div style='font-size:100%;'>
#     <a id='rename'></a>
#     <h1 style='color: black; font-weight: bold; font-family: Calibri;'>
#         <center> Renaming columns ⬇️ </center>
#     </h1>
# </div>

# In[38]:


dimProduct=dimProduct.rename(columns={'Product_PK':'FK_Product'} )
data['FK_Product']=pd.merge(data,dimProduct,on='FK_Product',how='left')['Product Name']


# In[ ]:





# <div style='font-size:100%;'>
#     <a id='cuf'></a>
#     <h1 style='color: black; font-weight: bold; font-family: Calibri;'>
#         <center> Chosing usefull features  ⬇️ </center>
#     </h1>
# </div>

# In[39]:



orders=data[['FK_Order','FK_Product']]


# In[40]:


data1=data.copy()


# <div style='font-size:100%;'>
#     <a id='splitting'></a>
#     <h1 style='color: black; font-weight: bold; font-family: Calibri;'>
#         <center> Spliting for a better using ⬇️ </center>
#     </h1>
# </div>

# In[41]:


data1['FK_Product'] = data['FK_Product'].str.split(',').str[0]


# In[ ]:





# In[42]:


orders=data1[['FK_Order','FK_Product']]  #group by


# <div style='font-size:100%;'>
#     <a id='listprod'></a>
#     <h1 style='color: black; font-weight: bold; font-family: Calibri;'>
#         <center> List of bought-together products⬇️ </center>
#     </h1>
# </div>

# In[43]:


dict={}
for j in  orders.index:
    if(not orders['FK_Order'][j] in dict):
         dict.update( [(orders['FK_Order'][j], [orders['FK_Product'][j]])] )
    else:
        dict[orders['FK_Order'][j]].append(orders['FK_Product'][j])

           


# In[44]:


listP=[]
for j in  dict.values():
    listP.append(j)


# In[ ]:





# <div style='font-size:100%;'>
#     <a id='Apriori'></a>
#     <h1 style='color: black; font-weight: bold; font-family: Calibri;'>
#         <center> Application of Apriori ⬇️ </center>
#     </h1>
# </div>

# In[45]:


association_rules = apriori(listP, min_support=0.00008, min_confidence=0.1, min_lift=1, min_length=2)
association_results = list(association_rules)


# <div style='font-size:100%;'>
#     <a id='Result'></a>
#     <h1 style='color: black; font-weight: bold; font-family: Calibri;'>
#         <center> Printing Apriori Results⬇️ </center>
#     </h1>
# </div>

# In[46]:


# Ce script sert à afficher la règle, le support, la confiance et le lift d'une manière claire


# In[47]:


result_list = []
for item in association_results:
    
    pair = item[0] 
    items = [x for x in pair]
    result_dict = {}
    result_dict['item_1'] = str(items[0])
    result_dict['item_2'] = str(items[1])
    result_list.append(result_dict)
df_apriori = pd.DataFrame(result_list)
    
   


# In[48]:


df_apriori


# In[ ]:




