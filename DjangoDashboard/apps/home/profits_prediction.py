
# In[1]:



from xgboost.sklearn import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
import statsmodels.formula.api as smf

from sklearn.model_selection import train_test_split
import warnings

import plotly.offline as pyoff
import plotly.graph_objs as go


# import Keras
import keras
from keras.layers import Dense
from keras.models import Sequential
from keras.layers import LSTM


# In[2]:


import pandas as pd
import numpy as np
# Libraries for data visualization

from tqdm.notebook import tqdm
import pygrametl



# In[3]:


import pyodbc
conn2 = pyodbc.connect('Driver={SQL Server Native Client 11.0};'
                      'Server=sql8004.site4now.net;'
                      'Database=db_a92253_innovision;'
                      'UID=db_a92253_innovision_admin;'
                      'PWD=innovision2022;'
                      )

cursor2 = pygrametl.ConnectionWrapper(connection=conn2)


# In[4]:


# Suppress all warnings
warnings.filterwarnings("ignore")


# In[5]:




pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 100)
pd.set_option('display.float_format', lambda x: '%.2f' % x)


# In[ ]:



# In[6]:


# Reading SQL query into a DataFrame

SQL_Query = pd.read_sql_query(
    '''select * FROM db_a92253_innovision.dbo.FactSales''', conn2)
dataFact = pd.DataFrame(SQL_Query)
SQL_Query = pd.read_sql_query(
    '''select DateKey,Date FROM db_a92253_innovision.dbo.DimDate''', conn2)
dimDate = pd.DataFrame(SQL_Query)





dimDate = dimDate.rename(columns={'DateKey': 'FK_Date_Order'})
dataFact['FK_Date_Order'] = pd.merge(
    dataFact, dimDate, on='FK_Date_Order', how='left')['Date']






# In[11]:


data = dataFact.copy()


# In[12]:




# In[13]:







# In[15]:


features = ['Quantity', 'Profit', 'Sales', 'Discount', 'Returned', 'Ship_Coast', 'Ship_Duration', 'Year', 'Month',
            'Day']






# In[17]:


target = ['Profit']












# In[20]:


data.drop(['FK_Customer', 'FK_Product', 'FK_Ship', 'FK_Date_Order',
          'FK_Date_Ship', 'FK_Order', 'FK_Localisation'], inplace=True, axis=1)



# In[22]:


# encoding "returned"
data['Returned'] = data['Returned'].replace(['NO'], '0')
data['Returned'] = data['Returned'].replace(['Yes'], '1')
data['Returned'] = data['Returned'].apply(int)






# In[71]:


dataFact['FK_Date_Order'] = pd.to_datetime(
    dataFact.FK_Date_Order, format='%Y-%m-%d')


# In[72]:


data = dataFact.copy()


# In[73]:


# represent month in date field as its first day
data['FK_Date_Order'] = data['FK_Date_Order'].dt.year.astype(
    'str') + '-' + data['FK_Date_Order'].dt.month.astype('str') + '-01'
data['FK_Date_Order'] = pd.to_datetime(data['FK_Date_Order'])


# removing the outliers from the feature "Profit"

# In[74]:


def remove_outlier(df, column):
    Q1 = np.percentile(df[column], 25, interpolation='midpoint')
    # Q2 = np.percentile(df[column], 50, interpolation = 'midpoint')
    Q3 = np.percentile(df[column], 75, interpolation='midpoint')
    IQR = Q3 - Q1
    low_lim = Q1 - 1.5 * IQR
    up_lim = Q3 + 1.5 * IQR
    print('*', column)
    print('low_limit is', low_lim)
    print('up_limit is', up_lim)
    print('\n')
    df = df[(df[column] < up_lim) & (df[column] > low_lim)]
    return df


# In[75]:


data = remove_outlier(data, 'Profit')


# In[76]:


data = data.groupby('FK_Date_Order').Profit.sum().reset_index()



# In[78]:


# create a new dataframe to model the difference
df_diff = data.copy()
# add previous sales to the next row
df_diff['prev_Profit'] = df_diff['Profit'].shift(1)
# drop the null values and calculate the difference
df_diff = df_diff.dropna()
df_diff['diff'] = (df_diff['Profit'] - df_diff['prev_Profit'])


# In[79]:


# In[80]:


# create dataframe for transformation from time series to supervised
df_supervised = df_diff.drop(['prev_Profit'], axis=1)

# adding lags
for inc in range(1, 24):
    field_name = 'lag_' + str(inc)
    df_supervised[field_name] = df_supervised['diff'].shift(inc)
# drop null values
df_supervised = df_supervised.dropna().reset_index(drop=True)




# In[82]:


# Import statsmodels.formula.api
# Define the regression formula
model = smf.ols(formula='diff ~ lag_1 + lag_2 + lag_3 +lag_4 + lag_5 + lag_6 + lag_7 + lag_8 + lag_9 + lag_10 + lag_11 + lag_12 + lag_13  + lag_14 + lag_15 + lag_16 + lag_17 + lag_18 + lag_19    ', data=df_supervised)
# Fit the regression
model_fit = model.fit()
# Extract the adjusted r-squared
regression_adj_rsq = model_fit.rsquared_adj



#         <center>XGBOOST Method</center>


# In[96]:


# import MinMaxScaler and create a new dataframe for LSTM model
df_model = df_supervised.drop(['FK_Date_Order', 'Profit'], axis=1)
# split train and test set
train_set, test_set = df_model[0:-6].values, df_model[-6:].values


# In[97]:


# apply Min Max Scaler
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(train_set)
# reshape training set
train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
train_set_scaled = scaler.transform(train_set)
# reshape test set
test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
test_set_scaled = scaler.transform(test_set)


# In[98]:


X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1].ravel()
X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1].ravel()


# In[99]:


# Run regression model
XG = XGBRegressor(n_estimators=200, learning_rate=0.5,
                  objective='reg:squarederror')

XG.fit(X_train, y_train)
y_pred = XG.predict(X_test)


# In[100]:


# reshape y_pred
y_pred = y_pred.reshape(y_pred.shape[0], 1, 1)
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
# rebuild test set for inverse transform
pred_test_set = []
for index in range(0, len(y_pred)):
    pred_test_set.append(np.concatenate(
        [y_pred[index], X_test[index]], axis=1))

# reshape pred_test_set
pred_test_set = np.array(pred_test_set)
pred_test_set = pred_test_set.reshape(
    pred_test_set.shape[0], pred_test_set.shape[2])
# inverse transform
pred_test_set_inverted = scaler.inverse_transform(pred_test_set)


# In[101]:


# create dataframe that shows the predicted Profit
result_list = []
Profit_dates = list(data[-7:].FK_Date_Order)
act_Profit = list(data[-7:].Profit)
for index in range(0, len(pred_test_set_inverted)):
    result_dict = {}
    result_dict['predicted_XGBOOST_Profit'] = int(
        pred_test_set_inverted[index][0] + act_Profit[index])
    result_dict['FK_Date_Order'] = Profit_dates[index+1]
    result_list.append(result_dict)
df_result_XGBOOST = pd.DataFrame(result_list)


# In[102]:


# merge with actual Profit dataframe
df_sales_pred = pd.merge(data, df_result_XGBOOST,
                         on='FK_Date_Order', how='left')


df_sales_pred['FK_Date_Order']= df_sales_pred['FK_Date_Order'].astype(str)
df_sales_pred['predicted_XGBOOST_Profit'] = df_sales_pred['predicted_XGBOOST_Profit'].fillna(df_sales_pred['Profit'])
df_sales_pred['predicted_XGBOOST_Profit']=df_sales_pred['predicted_XGBOOST_Profit'].round(decimals = 0)
df_sales_pred['Profit']=df_sales_pred['Profit'].round(decimals = 0)






#6 months predictions





data = dataFact.copy()


# In[73]:


# represent month in date field as its first day
data['FK_Date_Order'] = data['FK_Date_Order'].dt.year.astype(
    'str') + '-' + data['FK_Date_Order'].dt.month.astype('str') + '-01'
data['FK_Date_Order'] = pd.to_datetime(data['FK_Date_Order'])


# removing the outliers from the feature "Profit"

# In[74]:


def remove_outlier(df, column):
    Q1 = np.percentile(df[column], 25, interpolation='midpoint')
    # Q2 = np.percentile(df[column], 50, interpolation = 'midpoint')
    Q3 = np.percentile(df[column], 75, interpolation='midpoint')
    IQR = Q3 - Q1
    low_lim = Q1 - 1.5 * IQR
    up_lim = Q3 + 1.5 * IQR
    print('*', column)
    print('low_limit is', low_lim)
    print('up_limit is', up_lim)
    print('\n')
    df = df[(df[column] < up_lim) & (df[column] > low_lim)]
    return df


# In[75]:


data = remove_outlier(data, 'Profit')


# In[76]:


data = data.groupby('FK_Date_Order').Profit.sum().reset_index()
data.loc[len(data.index)] = ['2016-01-01', 10800.26] 
data.loc[len(data.index)] = ['2016-02-01', 9816.75] 
data.loc[len(data.index)] = ['2016-03-01', 13858.61] 
data.loc[len(data.index)] = ['2016-04-01', 19735.29] 
data.loc[len(data.index)] = ['2016-05-01', 10069.35] 
data.loc[len(data.index)] = ['2016-06-01', 14123.89] 


# In[78]:


# create a new dataframe to model the difference
df_diff = data.copy()
# add previous sales to the next row
df_diff['prev_Profit'] = df_diff['Profit'].shift(1)
# drop the null values and calculate the difference
df_diff = df_diff.dropna()
df_diff['diff'] = (df_diff['Profit'] - df_diff['prev_Profit'])


# In[79]:


# In[80]:


# create dataframe for transformation from time series to supervised
df_supervised = df_diff.drop(['prev_Profit'], axis=1)

# adding lags
for inc in range(1, 24):
    field_name = 'lag_' + str(inc)
    df_supervised[field_name] = df_supervised['diff'].shift(inc)
# drop null values
df_supervised = df_supervised.dropna().reset_index(drop=True)




# In[82]:


# Import statsmodels.formula.api
# Define the regression formula
model = smf.ols(formula='diff ~ lag_1 + lag_2 + lag_3 +lag_4 + lag_5 + lag_6 + lag_7 + lag_8 + lag_9 + lag_10 + lag_11 + lag_12 + lag_13 ', data=df_supervised)
# Fit the regression
model_fit = model.fit()
# Extract the adjusted r-squared
regression_adj_rsq = model_fit.rsquared_adj



#         <center>XGBOOST Method</center>


# In[96]:


# import MinMaxScaler and create a new dataframe for LSTM model
df_model = df_supervised.drop(['FK_Date_Order', 'Profit'], axis=1)
# split train and test set
train_set, test_set = df_model[0:-6].values, df_model[-6:].values


# In[97]:


# apply Min Max Scaler
scaler = MinMaxScaler(feature_range=(-1, 1))
scaler = scaler.fit(train_set)
# reshape training set
train_set = train_set.reshape(train_set.shape[0], train_set.shape[1])
train_set_scaled = scaler.transform(train_set)
# reshape test set
test_set = test_set.reshape(test_set.shape[0], test_set.shape[1])
test_set_scaled = scaler.transform(test_set)


# In[98]:


X_train, y_train = train_set_scaled[:, 1:], train_set_scaled[:, 0:1].ravel()
X_test, y_test = test_set_scaled[:, 1:], test_set_scaled[:, 0:1].ravel()


# In[99]:


# Run regression model
XG = XGBRegressor(n_estimators=200, learning_rate=0.5,
                  objective='reg:squarederror')

XG.fit(X_train, y_train)
y_pred = XG.predict(X_test)


# In[100]:


# reshape y_pred
y_pred = y_pred.reshape(y_pred.shape[0], 1, 1)
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
# rebuild test set for inverse transform
pred_test_set = []
for index in range(0, len(y_pred)):
    pred_test_set.append(np.concatenate(
        [y_pred[index], X_test[index]], axis=1))

# reshape pred_test_set
pred_test_set = np.array(pred_test_set)
pred_test_set = pred_test_set.reshape(
    pred_test_set.shape[0], pred_test_set.shape[2])
# inverse transform
pred_test_set_inverted = scaler.inverse_transform(pred_test_set)


# In[101]:


# create dataframe that shows the predicted Profit
result_list = []
Profit_dates = list(data[-7:].FK_Date_Order)
act_Profit = list(data[-7:].Profit)
for index in range(0, len(pred_test_set_inverted)):
    result_dict = {}
    result_dict['predicted_XGBOOST_Profit'] = int(
        pred_test_set_inverted[index][0] + act_Profit[index])
    result_dict['FK_Date_Order'] = Profit_dates[index+1]
    result_list.append(result_dict)
df_result_XGBOOST = pd.DataFrame(result_list)

# In[102]:


# merge with actual Profit dataframe
df_sales_pred2 = pd.merge(data, df_result_XGBOOST,
                         on='FK_Date_Order', how='left')


df_sales_pred2['FK_Date_Order']= df_sales_pred2['FK_Date_Order'].astype(str)
df_sales_pred2= df_sales_pred2.dropna()
df_sales_pred2['predicted_XGBOOST_Profit']=df_sales_pred2['predicted_XGBOOST_Profit'].round(decimals = 0)
df_sales_pred2['Profit']=df_sales_pred2['Profit'].round(decimals = 0)