#!/usr/bin/env python
# coding: utf-8

# ### Project Title:

# # PRODUCT DEMAND FORECASTING

# # Installations

# In[1]:


'''
%pip install pyodbc
!pip install python-dotenv
!pip install plotly
!pip install statsmodels
!pip install tensorflow
!pip install --upgrade tensorflow
'''


# # Importation

# In[2]:


import pyodbc
from dotenv import dotenv_values
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import seaborn as sns
import plotly.express as px

from statsmodels.tsa.stattools import adfuller
from scipy.stats import ttest_ind
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_log_error

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.model_selection import RandomizedSearchCV
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, GRU, Dense, Dropout, Dropout, Bidirectional, BatchNormalization
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import pickle
import joblib

import warnings
warnings.filterwarnings("ignore")


# # Data Loading and Exploration

# ## Loading the datasets

# In[3]:


import pandas as pd
oil_df = pd.read_csv(r'oil.csv')
holidays_events_df = pd.read_csv(r'holidays_events.csv')
stores_df = pd.read_csv(r'stores.csv')


# In[4]:


oil_df.head()


# In[5]:


holidays_events_df.head()


# In[6]:


stores_df.head()


# In[7]:


transactions_df = pd.read_csv(r"transactions.csv")
transactions_df.head()


# ## Loading the Train data

# In[8]:


train_df = pd.read_csv(r"train.csv")
train_df.head()


# ## Loading the Test data

# In[9]:


test_df = pd.read_csv(r"test.csv")
test_df.head()


# # EDA, Data Preprocessing & Cleaning

# ## i. Shape of The Datasets

# In[10]:


print(f"Train Dataset: {train_df.shape}")
print(f"Test Datasets: {test_df.shape}")


# In[11]:


print(f"Holiday Events Dataset: {holidays_events_df.shape}")
print(f"Oil Dataset: {oil_df.shape}")
print(f"Stores Dataset: {stores_df.shape}")
print(f"Transactions Dataset: {transactions_df.shape}")


# ## ii. Column Information of The Datasets

# In[12]:


def show_column_info(dataset_name, dataset):
    print(f"Data types for the {dataset_name} dataset:")
    print(dataset.info())
    print('==='*14)


# In[13]:


show_column_info('Train', train_df)
print()
show_column_info('Test', test_df)


# In[14]:


show_column_info('Holiday events', holidays_events_df)
print()
show_column_info('Oil', oil_df)
print()
show_column_info('Stores', stores_df)
print()
show_column_info('Transactions', transactions_df)


# ## iii. Transforming the 'date' column to datetime format

# In[15]:


train_df['date'] = pd.to_datetime(train_df['date'])
test_df['date'] = pd.to_datetime(test_df['date'])

holidays_events_df['date'] = pd.to_datetime(holidays_events_df['date'])
oil_df['date'] = pd.to_datetime(oil_df['date'])
transactions_df['date'] = pd.to_datetime(transactions_df['date'])


# In[16]:


print('Date Column Data Type After Transformation:')
print('==='*14)
print("Train dataset:", train_df['date'].dtype)
print("Test dataset:", test_df['date'].dtype)
print("Holiday Events dataset:", holidays_events_df['date'].dtype)
print("Oil dataset:", oil_df['date'].dtype)
print("Transactions dataset:", transactions_df['date'].dtype)


# ## iv. Summary statistics of the datasets

# In[17]:


datasets = {'train': train_df, 'test': test_df, 'holiday events': holidays_events_df, 'oil': oil_df, 'stores': stores_df, 'transactions': transactions_df}
for name, data in datasets.items():
    print(f"{name.capitalize()} dataset summary statistics:")
    print('---'*15)
    print(data.describe())
    print('==='*20)
    print()


# ## v. Checking for Missing Values in The Datasets

# In[18]:


datasets = {'train': train_df, 'test': test_df, 'holiday events': holidays_events_df, 'oil': oil_df, 'stores': stores_df, 'transactions': transactions_df, }
def show_missing_values(datasets):
    for name, data in datasets.items():
        print(f"Missing values in the {name.capitalize()} dataset:")
        print(data.isnull().sum())
        print('===' * 18)
        print()

show_missing_values(datasets)


# ### Handling the Missing Values in the 'dcoilwtico' (daily crude oil prices) of the Oil Dataset.
# 
# 

# In[19]:


fig = px.line(oil_df, x='date', y='dcoilwtico')
fig.update_layout(title='Trend of Oil Prices Over Time', title_x=0.5, xaxis_title='Date', yaxis_title='Oil Price')
fig.update_xaxes(rangeslider_visible=True)
fig.show()


# In[20]:


oil_df['dcoilwtico'] = oil_df['dcoilwtico'].fillna(method='backfill')


# In[21]:


missing_values_after = oil_df['dcoilwtico'].isnull().sum()
missing_values_after


# In[22]:


#check
fig = px.line(oil_df, x='date', y='dcoilwtico')
fig.update_layout(title='Trend of Oil Prices Over Time', title_x=0.5, xaxis_title='Date', yaxis_title='Oil Price')
fig.update_xaxes(rangeslider_visible=True)
fig.show()


#  ## vi. Checking for the completeness of the 'date' column in the Train Dataset

# In[23]:


min_date = train_df['date'].min()
max_date = train_df['date'].max()
expected_dates = pd.date_range(start=min_date, end=max_date)
missing_dates = expected_dates[~expected_dates.isin(train_df['date'])]
if len(missing_dates) == 0:
    print("The train dataset is complete. It includes all the required dates.")
else:
    print("The train dataset is incomplete. The following dates are missing:")
    print(missing_dates)


# In[24]:


missing_dates = pd.Index(['2013-12-25', '2014-12-25', '2015-12-25', '2016-12-25'], dtype='datetime64[ns]')
missing_data = pd.DataFrame({'date': missing_dates})
# ignore_index=True... ensures a new index is assigned to the resulting DataFrame
train_df = pd.concat([train_df, missing_data], ignore_index=True)
train_df.sort_values('date', inplace=True)


# In[25]:


#check
min_date = train_df['date'].min()
max_date = train_df['date'].max()
expected_dates = pd.date_range(start=min_date, end=max_date)
missing_dates = expected_dates[~expected_dates.isin(train_df['date'])]
if len(missing_dates) == 0:
    print("The train dataset is complete. It includes all the required dates.")
else:
    print("The train dataset is incomplete. The following dates are missing:")
    print(missing_dates)


# ## vi. Merging The Train Dataset with the Stores, Transactions, Holiday Events and Oil Dataset

# In[26]:


#inner merge() function
merged_df1 = train_df.merge(stores_df, on='store_nbr', how='inner')
merged_df2 = merged_df1.merge(transactions_df, on=['date', 'store_nbr'], how='inner')
merged_df3 = merged_df2.merge(holidays_events_df, on='date', how='inner')
merged_df = merged_df3.merge(oil_df, on='date', how='inner')
merged_df.head()


# In[27]:


merged_df.info()


# In[28]:


print("Unique values of 'type_x':")
print(merged_df['type_x'].unique())
print()
print("Unique values of 'type_y':")
print(merged_df['type_y'].unique())


# In[29]:


merged_df = merged_df.rename(columns={"type_x": "store_type", "type_y": "holiday_type"})
merged_df.head()


# In[30]:


#ss and T for better readability
merged_df.describe().T


# ## viii. Checking for Missing Values in The Datasets

# In[31]:


missing_values = merged_df.isnull().sum()
missing_values


# ## ix. Checking for Duplicate Values in The Datasets

# In[32]:


duplicate_rows_merged = merged_df.duplicated()
duplicate_rows_merged.sum()


# In[33]:


duplicate_rows_test = test_df.duplicated()
duplicate_rows_test.sum()


# ## x. Save the merged dataset in a new CSV file to be used in Visualization

# In[34]:


merged_df.to_csv('Visualization_Data.csv', index=False)
merged_df.head()


# In[35]:


merged_df_copy = merged_df.copy()
merged_df_copy.info()


# # Data Visualization and Analysis

# ## SALES

# ## a. Distribution of the 'sales' variable:

# In[36]:


plt.boxplot(merged_df['sales'])
plt.ylabel('Sales')
plt.title('Boxplot of Sales')
plt.show()


# ## b. Trend of sales over time.

# In[37]:


#group by date, calc tot sales
daily_sales = merged_df.groupby('date')['sales'].sum().reset_index()

fig = px.line(daily_sales, x='date', y='sales')
fig.update_xaxes(rangeslider_visible=True)
fig.update_layout(title='Trend of Sales Over Time', title_x=0.5)
fig.show()


# ## c. Total Count of Sales by Store Type

# In[38]:


sns.set_palette("viridis")
store_type_counts = merged_df['store_type'].value_counts()
store_type_sales = merged_df.groupby('store_type')['sales'].sum()

plt.figure(figsize=(8, 6))
sns.barplot(x=store_type_counts.index, y=store_type_counts.values)
plt.xlabel('Store Type')
plt.ylabel('Count')
plt.title('Total Count by Store Type')
plt.show()


# ## d. Total Amount in Sales by Store Type

# In[39]:


store_type_sales = store_type_sales.sort_values(ascending=False)

plt.figure(figsize=(8, 8))
sns.barplot(x=store_type_sales.index, y=store_type_sales.values, order=store_type_sales.index, palette="viridis")
plt.xlabel('Store Type')
plt.ylabel('Total Sales')
plt.title('Total Sales by Store Type')
plt.show()


# ## e. Average Sales by City

# In[40]:


average_sales_by_city = merged_df.groupby('city')['sales'].mean()
average_sales_by_city = average_sales_by_city.sort_values(ascending=True)
colors = cm.viridis(np.linspace(0, 1, len(average_sales_by_city)))
plt.figure(figsize=(8, 8))
plt.barh(average_sales_by_city.index, average_sales_by_city.values, color=colors)
plt.xlabel('Average Sales')
plt.ylabel('City')
plt.title('Average Sales by City')
plt.show()


# ## TRANSACTIONS

# ## a. Distribution of the 'transactions' variable:

# In[41]:


plt.hist(merged_df['transactions'], bins=20)
plt.xlabel('Transactions')
plt.ylabel('Frequency')
plt.title('Distribution of Transactions')
plt.show()


# ## b. Average Sales by State

# In[42]:


average_sales_by_state = merged_df.groupby('state')['sales'].mean()
average_sales_by_state = average_sales_by_state.sort_values(ascending=True)
plt.figure(figsize=(8, 8))
plt.barh(average_sales_by_state.index, average_sales_by_state.values, color=colors)
plt.xlabel('Average Sales')
plt.ylabel('State')
plt.title('Average Sales by State')
plt.xticks(rotation=45)
plt.show()


# ## DAILY OIL PRICE

# ## a. Distribution of the 'Daily Oil Price' variable:

# In[43]:


plt.hist(merged_df['dcoilwtico'], bins=20)
plt.xlabel('Oil Price')
plt.ylabel('Frequency')
plt.title('Distribution of Oil Price')
plt.show()


# ## b. Trend of Daily Crude oil Prices Over Time

# In[44]:


fig = px.line(oil_df, x='date', y='dcoilwtico')
fig.update_layout(title='Trend of Oil Prices Over Time', title_x=0.5, xaxis_title='Date', yaxis_title='Oil Price')
fig.show()


# ## Relationship between sales and transactions.

# In[45]:


sns.scatterplot(x='transactions', y='sales', data=merged_df)
plt.xlabel('Transactions')
plt.ylabel('Sales')
plt.title('Relationship between Sales and Transactions')
plt.show()


# ## Correlation matrix of numerical variables

# In[46]:


numerical_vars = ['sales', 'transactions', 'dcoilwtico']
corr_matrix = merged_df[numerical_vars].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()


# ## Scatter Plot Marrix of numerical Variables

# In[47]:


numerical_vars = ['sales', 'transactions', 'dcoilwtico']
sns.pairplot(merged_df[numerical_vars])
plt.show()


# # Feature Enginering

# ## Train Dataset

# ### Extracting Date Components (Day, Month, Year and Day of The Week).

# In[48]:


merged_df_copy['date'] = pd.to_datetime(merged_df_copy['date'])
merged_df_copy['year'] = merged_df_copy['date'].dt.year
merged_df_copy['month'] = merged_df_copy['date'].dt.month
merged_df_copy['day'] = merged_df_copy['date'].dt.day
merged_df_copy.head()


# ### Dropping Unneccessary Columns in The Merged and Test Datasets (as it is not needed for the analysis)

# In[49]:


columns_to_drop = ['date','id', 'locale', 'locale_name', 'description', 'store_type', 'transferred', 'state']
merged_df_copy = merged_df_copy.drop(columns=columns_to_drop)

merged_df_copy.head()


# ### Product Categorization Based on Families

# In[50]:


unique_families = merged_df_copy['family'].unique()
unique_families


# In[51]:


#define category lists for each product category
food_families = ['BEVERAGES', 'BREAD/BAKERY', 'FROZEN FOODS', 'MEATS', 'PREPARED FOODS', 'DELI','PRODUCE', 'DAIRY','POULTRY','EGGS','SEAFOOD']
home_families = ['HOME AND KITCHEN I', 'HOME AND KITCHEN II', 'HOME APPLIANCES']
clothing_families = ['LINGERIE', 'LADYSWARE']
grocery_families = ['GROCERY I', 'GROCERY II']
stationery_families = ['BOOKS', 'MAGAZINES','SCHOOL AND OFFICE SUPPLIES']
cleaning_families = ['HOME CARE', 'BABY CARE','PERSONAL CARE']
hardware_families = ['PLAYERS AND ELECTRONICS','HARDWARE']

#categorize the 'family' column based on the product categories
merged_df_copy['family'] = np.where(merged_df_copy['family'].isin(food_families), 'FOODS', merged_df_copy['family'])
merged_df_copy['family'] = np.where(merged_df_copy['family'].isin(home_families), 'HOME', merged_df_copy['family'])
merged_df_copy['family'] = np.where(merged_df_copy['family'].isin(clothing_families), 'CLOTHING', merged_df_copy['family'])
merged_df_copy['family'] = np.where(merged_df_copy['family'].isin(grocery_families), 'GROCERY', merged_df_copy['family'])
merged_df_copy['family'] = np.where(merged_df_copy['family'].isin(stationery_families), 'STATIONERY', merged_df_copy['family'])
merged_df_copy['family'] = np.where(merged_df_copy['family'].isin(cleaning_families), 'CLEANING', merged_df_copy['family'])
merged_df_copy['family'] = np.where(merged_df_copy['family'].isin(hardware_families), 'HARDWARE', merged_df_copy['family'])
merged_df_copy.head()


# ### Feature Scaling

# In[52]:


#scaling Numeric Variables (Min-Max Scaling)
scaler = StandardScaler()
num_cols = ['sales', 'transactions', 'dcoilwtico']
merged_df_copy[num_cols] = scaler.fit_transform(merged_df_copy[num_cols])
merged_df_copy.head()


# ### Encoding The Categorical Variables

# In[53]:


categorical_columns = ["family", "city", "holiday_type"]
encoder = OneHotEncoder()
one_hot_encoded_data = encoder.fit_transform(merged_df_copy[categorical_columns])
column_names = encoder.get_feature_names_out(categorical_columns)
merged_df_encoded = pd.DataFrame(one_hot_encoded_data.toarray(), columns=column_names)
merged_df_encoded = pd.concat([merged_df_copy, merged_df_encoded], axis=1)
merged_df_encoded.drop(categorical_columns, axis=1, inplace=True)
merged_df_encoded.head()


# ## Test_df

# In[54]:


#extracting date components
test_df['date'] = pd.to_datetime(test_df['date'])
test_df['year'] = test_df['date'].dt.year
test_df['month'] = test_df['date'].dt.month
test_df['day'] = test_df['date'].dt.day
test_df.head()


# In[55]:


test_df.head()


# In[56]:


#drop unnecessary columns
columns_to_drop = ['date', 'id']
test_df = test_df.drop(columns=columns_to_drop)
test_df.head()


# In[57]:


#product categorization based on families
food_families = ['BEVERAGES', 'BREAD/BAKERY', 'FROZEN FOODS', 'MEATS', 'PREPARED FOODS', 'DELI', 'PRODUCE', 'DAIRY', 'POULTRY', 'EGGS', 'SEAFOOD']
home_families = ['HOME AND KITCHEN I', 'HOME AND KITCHEN II', 'HOME APPLIANCES']
clothing_families = ['LINGERIE', 'LADYSWARE']
grocery_families = ['GROCERY I', 'GROCERY II']
stationery_families = ['BOOKS', 'MAGAZINES', 'SCHOOL AND OFFICE SUPPLIES']
cleaning_families = ['HOME CARE', 'BABY CARE', 'PERSONAL CARE']
hardware_families = ['PLAYERS AND ELECTRONICS', 'HARDWARE']

test_df['family'] = np.where(test_df['family'].isin(food_families), 'FOODS', test_df['family'])
test_df['family'] = np.where(test_df['family'].isin(home_families), 'HOME', test_df['family'])
test_df['family'] = np.where(test_df['family'].isin(clothing_families), 'CLOTHING', test_df['family'])
test_df['family'] = np.where(test_df['family'].isin(grocery_families), 'GROCERY', test_df['family'])
test_df['family'] = np.where(test_df['family'].isin(stationery_families), 'STATIONERY', test_df['family'])
test_df['family'] = np.where(test_df['family'].isin(cleaning_families), 'CLEANING', test_df['family'])
test_df['family'] = np.where(test_df['family'].isin(hardware_families), 'HARDWARE', test_df['family'])


# In[58]:


#encode categorical vars
categorical_columns = ["family"]
encoder = OneHotEncoder()
one_hot_encoded_data = encoder.fit_transform(test_df[categorical_columns])
column_names = encoder.get_feature_names_out(categorical_columns)
test_df_encoded = pd.DataFrame(one_hot_encoded_data.toarray(), columns=column_names)
test_df_encoded = pd.concat([test_df, test_df_encoded], axis=1)
test_df_encoded.drop(categorical_columns, axis=1, inplace=True)
test_df_encoded.head()


# # Modeling

# ## Data Splitting

# In[59]:


train_set = merged_df_encoded.loc[merged_df_encoded['year'].isin([2013, 2014, 2015, 2016])]
eval_set = merged_df_encoded.loc[merged_df_encoded['year'].isin([2013, 2014])]


# In[60]:


train_set.head()


# In[61]:


train_set.shape


# In[62]:


eval_set.head()


# In[63]:


eval_set.shape


# In[64]:


eval_set.head()


# In[65]:


#separate the target variable and features for training and testing
X_train = train_set.drop('sales', axis=1)
y_train = train_set['sales']

X_eval = eval_set.drop('sales', axis=1)
y_eval = eval_set['sales']


# In[66]:


X_train


# In[67]:


y_train


# In[68]:


X_eval


# In[69]:


y_eval


# In[70]:


results_df = pd.DataFrame(columns=['Model', 'RMSLE', 'RMSE', 'MSE', 'MAE'])


# In[71]:


import pandas as pd
import matplotlib.pyplot as plt
X_eval['transactions'].plot.box(title='Boxplot of transactions')
plt.show()


# In[72]:


import pandas as pd
import matplotlib.pyplot as plt
X_train['transactions'].plot.box(title='Boxplot of transactions')
plt.show()


# In[73]:


print(5)


# ## Model 1. Linear Regression

# In[74]:


'''
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
'''


# In[75]:


'''
with open('LR_model.pkl','wb') as f:
    pickle.dump(lr_model,f)
'''


# In[76]:


with open('LR_model.pkl','rb') as f:
    lr_model1=pickle.load(f)


# In[77]:


lr_predictions = lr_model1.predict(X_eval)
#metrics
lr_mse = mean_squared_error(y_eval, lr_predictions)
lr_mae = mean_absolute_error(y_eval, lr_predictions)
#abs on y_eval and lr_predictions
y_eval_abs = np.abs(y_eval)
lr_predictions_abs = np.abs(lr_predictions)
#RMSLE
lr_rmsle = np.sqrt(mean_squared_log_error(y_eval_abs, lr_predictions_abs))
results_lr = pd.DataFrame({'Model': ['Linear Regression'],
                            'RMSLE': [lr_rmsle],
                            'RMSE': [np.sqrt(lr_mse)],
                            'MSE': [lr_mse],
                            'MAE': [lr_mae]}).round(3)
results_lr


# ## Model 2. ARIMA

# In[78]:


'''
p = 1
d = 0
q = 0

arima_model = ARIMA(y_train, order=(p, d, q))
arima_model_fit = arima_model.fit()
'''


# In[79]:


'''
with open('ARIMA_model.pkl','wb') as f:
    pickle.dump(arima_model_fit,f)
'''


# In[80]:


with open('ARIMA_model.pkl','rb') as f:
    arima_model1=pickle.load(f)


# In[81]:


arima_predictions = arima_model1.predict(start=len(y_train), end=len(y_train) + len(X_eval) - 1)

arima_mse = mean_squared_error(y_eval, arima_predictions)
arima_rmse = np.sqrt(arima_mse)
y_eval_abs = np.abs(y_eval)
arima_predictions_abs = np.abs(arima_predictions)
arima_mae = mean_absolute_error(y_eval, arima_predictions)
arima_rmsle = np.sqrt(mean_squared_log_error(y_eval_abs, arima_predictions_abs))

results_arima = pd.DataFrame({'Model': ['ARIMA'],
                            'RMSLE': [arima_rmsle],
                            'RMSE': [np.sqrt(arima_mse)],
                            'MSE': [arima_mse],
                            'MAE': [arima_mae]}).round(3)
results_arima


# 
# ## Model 3. XGBoost Regressor

# In[82]:


import xgboost as xgb


# In[83]:


'''
xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=6, learning_rate=0.1)
xgb_model.fit(X_train, y_train)
'''


# In[84]:


'''
with open('XGB_model.pkl','wb') as f:
    pickle.dump(xgb_model,f)
'''


# In[85]:


with open('XGB_model.pkl','rb') as f:
    xgb_model1=pickle.load(f)


# In[86]:


xgb_preds = xgb_model1.predict(X_eval)
xgb_rmsle = np.sqrt(mean_squared_log_error(np.abs(y_eval), np.abs(xgb_preds)))
xgb_results = pd.DataFrame({'Model': ['XGBoost'], 'RMSLE': [xgb_rmsle], 'RMSE': [np.sqrt(mean_squared_error(y_eval, xgb_preds))], 'MSE': [mean_squared_error(y_eval, xgb_preds)], 'MAE': [mean_absolute_error(y_eval, xgb_preds)]}).round(1)


# In[87]:


xgb_results


# ##  Model 4. Random Forest

# In[88]:


'''
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
'''


# In[89]:


'''
with open('RF_model.pkl','wb') as f:
    pickle.dump(rf_model,f)
'''


# In[90]:


with open('RF_model.pkl','rb') as f:
    rf_model1=pickle.load(f)


# In[91]:


rf_predictions = rf_model1.predict(X_eval)


# In[92]:


# Calculate metrics
rf_mse = mean_squared_error(y_eval, rf_predictions)
rf_mae = mean_absolute_error(y_eval, rf_predictions)


# In[93]:


# Apply the absolute value function to both y_eval and rf_predictions
y_eval_abs = abs(y_eval)
rf_predictions_abs = abs(rf_predictions)


# In[94]:


# Calculate the Root Mean Squared Logarithmic Error (RMSLE)
rf_rmsle = np.sqrt(mean_squared_log_error(y_eval_abs, rf_predictions_abs))

# Create a DataFrame to store results for Random Forest
results_rf = pd.DataFrame({'Model': ['Random Forest'],
                            'RMSLE': [rf_rmsle],
                            'RMSE': [np.sqrt(rf_mse)],
                            'MSE': [rf_mse],
                            'MAE': [rf_mae]}).round(1)

# Print the results_rf dataframe
results_rf


# ## Model 5. GB Regressor

# In[95]:


'''
gb_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
gb_model.fit(X_train, y_train)
'''


# In[96]:


'''
with open('GB_model.pkl','wb') as f:
    pickle.dump(gb_model,f)
'''


# In[97]:


with open('GB_model.pkl','rb') as f:
    gb_model1=pickle.load(f)


# In[98]:


gb_predictions = gb_model1.predict(X_eval)


# In[99]:


# Calculate metrics
gb_mse = mean_squared_error(y_eval, gb_predictions)
gb_mae = mean_absolute_error(y_eval, gb_predictions)


# In[100]:


# Apply the absolute value function to both y_eval and gb_predictions
y_eval_abs = np.abs(y_eval)
gb_predictions_abs = np.abs(gb_predictions)


# In[101]:


# Calculate the Root Mean Squared Logarithmic Error (RMSLE)
gb_rmsle = np.sqrt(mean_squared_log_error(y_eval_abs, gb_predictions_abs))

# Create a DataFrame to store results for Gradient Boosting
results_gb = pd.DataFrame({'Model': ['Gradient Boosting'],
                            'RMSLE': [gb_rmsle],
                            'RMSE': [np.sqrt(gb_mse)],
                            'MSE': [gb_mse],
                            'MAE': [gb_mae]}).round(3)

# Print the results_gb dataframe
results_gb


# ## Model 6. Cat Boost

# In[102]:


'''from catboost import CatBoostRegressor'''


# In[103]:


'''
cat_model = CatBoostRegressor(iterations=500,
                              learning_rate=0.1,
                              depth=6,
                              verbose=0,
                              random_state=42)

cat_model.fit(X_train, y_train)
'''


# In[104]:


'''
with open('CAT_model.pkl','wb') as f:
    pickle.dump(cat_model,f)
'''


# In[105]:


with open('CAT_model.pkl','rb') as f:
    cat_model1=pickle.load(f)


# In[106]:


cat_predictions = cat_model1.predict(X_eval)


# In[107]:


# Calculate metrics
cat_mse = mean_squared_error(y_eval, cat_predictions)
cat_mae = mean_absolute_error(y_eval, cat_predictions)


# In[108]:


# Apply the absolute value function to both y_eval and cat_predictions
y_eval_abs = np.abs(y_eval)
cat_predictions_abs = np.abs(cat_predictions)


# In[109]:


# Calculate the Root Mean Squared Logarithmic Error (RMSLE)
cat_rmsle = np.sqrt(mean_squared_log_error(y_eval_abs, cat_predictions_abs))

# Create a DataFrame to store results for CatBoost
results_cat = pd.DataFrame({'Model': ['CatBoost'],
                            'RMSLE': [cat_rmsle],
                            'RMSE': [np.sqrt(cat_mse)],
                            'MSE': [cat_mse],
                            'MAE': [cat_mae]}).round(1)

# Print the results_cat dataframe
results_cat


# ## Model 7. Model Stacking

# What is Model Stacking?
# 
# --- Model stacking(a type of hybrid modeling) is a powerful ensemble technique where multiple models (base learners) make predictions, and a meta-model learns to combine these predictions for better performance.
# 
# 1. Base Learners (Level 0): Train different models (e.g., XGBoost, CatBoost, Random Forest) on training data.
# 
# 2. Meta Learner (Level 1): A simple model (e.g., Linear Regression or CatBoost) trained on predictions made by base models (on validation set).
# 
# 3. During prediction: Use base models to generate predictions on test set and feed them into the meta-model for the final output.

# How We'll Do It:
# 1. Split training data into k-folds.
# 2. For each base model, generate out-of-fold predictions.
# 3. Train a meta-model using those out-of-fold predictions.
# 4. Use trained base models to predict on the evaluation set, and pass those to the meta-model for final predictions.
# 
# 

# ## Model Stacking (XGBoost, CatBoost, Random Forest → Linear Regression)

# In[110]:


'''from sklearn.model_selection import KFold'''


# #### Initialize models
# xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=6, learning_rate=0.1)
# 
# cat_model = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, verbose=0)
# 
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

# In[111]:


'''
# Prepare containers
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

meta_train = np.zeros((X_train.shape[0], 3))  # 3 base models
meta_test = np.zeros((X_eval.shape[0], 3))

for i, model in enumerate([xgb_model1, cat_model1, rf_model1]):
    test_fold_preds = []
    for train_idx, valid_idx in kf.split(X_train):
        X_t, X_val = X_train.iloc[train_idx], X_train.iloc[valid_idx]
        y_t, y_val = y_train.iloc[train_idx], y_train.iloc[valid_idx]

        model.fit(X_t, y_t)
        meta_train[valid_idx, i] = model.predict(X_val)
        test_fold_preds.append(model.predict(X_eval))

    # Average predictions from each fold
    meta_test[:, i] = np.mean(test_fold_preds, axis=0)
'''


# In[112]:


'''
import joblib

# Save the meta feature arrays
joblib.dump(meta_train, 'META_train.pkl')
joblib.dump(meta_test, 'META_test.pkl')
'''


# In[113]:


import joblib

# Load saved meta features
meta_train1 = joblib.load('META_train.pkl')
meta_test1 = joblib.load('META_test.pkl')


# In[114]:


'''
# Train meta-learner
meta_model = LinearRegression()
meta_model.fit(meta_train1, y_train)
'''


# In[115]:


'''
with open('META_model.pkl','wb') as f:
    pickle.dump(meta_model,f)
'''


# In[116]:


with open('META_model.pkl','rb') as f:
    meta_model1=pickle.load(f)


# In[117]:


# Final stacked predictions
stacked_preds = meta_model1.predict(meta_test1)

# Metrics
stacked_rmsle = np.sqrt(mean_squared_log_error(np.abs(y_eval), np.abs(stacked_preds)))
stacked_rmse = np.sqrt(mean_squared_error(y_eval, stacked_preds))
stacked_mse = mean_squared_error(y_eval, stacked_preds)
stacked_mae = mean_absolute_error(y_eval, stacked_preds)

# Results
results_stacked = pd.DataFrame({
    'Model': ['Stacked (XGB + CAT + RF → Linear)'],
    'RMSLE': [stacked_rmsle],
    'RMSE': [stacked_rmse],
    'MSE': [stacked_mse],
    'MAE': [stacked_mae]
}).round(2)

results_stacked


# 

# In[ ]:





# # Hyperparameter Tuning

# ### Tune Base Learners Individually

# In[118]:


'''
from sklearn.model_selection import RandomizedSearchCV

xgb_params = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 4, 6],
    'learning_rate': [0.01, 0.05, 0.1],
}

xgb_search = RandomizedSearchCV(
    xgb.XGBRegressor(objective='reg:squarederror'), 
    xgb_params, 
    n_iter=5, 
    cv=3, 
    random_state=42, 
    n_jobs=-1
)
xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_
'''


# In[119]:


'''
with open('best_XGB.pkl', 'wb') as f:
    pickle.dump(best_xgb, f)
'''


# In[120]:


# Load tuned model
with open('best_XGB.pkl', 'rb') as f:
    best_xgb1 = pickle.load(f)


# In[121]:


'''
cat_params = {
    'iterations': [100, 200],
    'learning_rate': [0.05, 0.1],
    'depth': [4, 6, 8],
}

cat_search = RandomizedSearchCV(
    CatBoostRegressor(verbose=0, random_state=42), 
    cat_params, 
    n_iter=5, 
    cv=3, 
    random_state=42, 
    n_jobs=-1
)
cat_search.fit(X_train, y_train)
best_cat = cat_search.best_estimator_
'''


# In[122]:


'''
# Save tuned model
import pickle

with open('best_CAT.pkl', 'wb') as f:
    pickle.dump(best_cat, f)
'''


# In[123]:


# Load tuned model
with open('best_CAT.pkl', 'rb') as f:
    best_cat1 = pickle.load(f)


# In[124]:


'''
rf_params = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
}

rf_search = RandomizedSearchCV(
    RandomForestRegressor(random_state=42), 
    rf_params, 
    n_iter=5, 
    cv=3, 
    random_state=42, 
    n_jobs=-1
)
rf_search.fit(X_train, y_train)
best_rf = rf_search.best_estimator_
'''


# In[125]:


'''
# Save tuned model
import pickle

with open('best_RF.pkl', 'wb') as f:
    pickle.dump(best_rf, f)
'''


# In[126]:


# Load tuned model
with open('best_RF.pkl', 'rb') as f:
    best_rf1 = pickle.load(f)


# ### Tune the Stacked Model (Meta-Learner)

# In[127]:


'''
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
'''


# In[128]:


'''
# Define base models
estimators = [
    ('xgb', best_xgb1),
    ('cat', best_cat1),
    ('rf', best_rf1)
]
'''


# In[129]:


'''
# Try Ridge first, then GBRegressor if Ridge underperforms
meta_learner = make_pipeline(
    StandardScaler(), Ridge(alpha=1.0)
)
'''


# In[130]:


'''
# Define stacking regressor
stack = StackingRegressor(
    estimators=estimators,
    final_estimator=meta_learner,
    passthrough=True,
    n_jobs=-1
)
'''


# In[131]:


'''
# Hyperparameter grid
meta_params = {
    'final_estimator__ridge__alpha': [0.01, 0.1, 1.0, 10.0]
}

stack_search = GridSearchCV(
    stack,
    param_grid=meta_params,
    cv=3,
    scoring='neg_root_mean_squared_log_error',  # More intuitive error
    n_jobs=-1,
    verbose=1
)
'''


# In[132]:


'''
stack_search.fit(X_train, y_train)
'''


# In[133]:


'''joblib.dump(stack_search, 'STACK_search.pkl')'''


# In[134]:


stack_search1 = joblib.load('STACK_search.pkl')
print(stack_search1.best_params_)


# In[135]:


best_stacked_model = stack_search1.best_estimator_


# In[136]:


# Save tuned model
import pickle

with open('best_STACKED_MODEL.pkl', 'wb') as f:
    pickle.dump(best_stacked_model, f)


# In[137]:


# Load tuned model
with open('best_STACKED_MODEL.pkl', 'rb') as f:
    best_stacked_model1 = pickle.load(f)


# ### Predict and Evaluate on Validation/Test Set

# In[138]:


stacked_preds = best_stacked_model1.predict(X_eval)

stacked_rmsle = np.sqrt(mean_squared_log_error(np.abs(y_eval), np.abs(stacked_preds)))
stacked_rmse = np.sqrt(mean_squared_error(y_eval, stacked_preds))
stacked_mse = mean_squared_error(y_eval, stacked_preds)
stacked_mae = mean_absolute_error(y_eval, stacked_preds)

results_stacked = pd.DataFrame({
    'Model': ['Stacked Tuned Model'],
    'RMSLE': [stacked_rmsle],
    'RMSE': [stacked_rmse],
    'MSE': [stacked_mse],
    'MAE': [stacked_mae]
}).round(3)

results_stacked


# # Further tuning using optuna

# In[139]:


'''
pip install optuna
'''


# In[152]:


import optuna
import numpy as np
from sklearn.ensemble import StackingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_squared_log_error, make_scorer
from xgboost import XGBRegressor
from catboost import CatBoostRegressor


# In[153]:


# Define RMSLE scorer
def rmsle_scorer(y_true, y_pred):
    return np.sqrt(mean_squared_log_error(np.abs(y_true), np.abs(y_pred)))

neg_rmsle_scorer = make_scorer(rmsle_scorer, greater_is_better=False)


# In[154]:


# Define base models
best_xgb2 = XGBRegressor(objective='reg:squarederror', n_estimators=100, max_depth=6, learning_rate=0.1)
best_cat2 = CatBoostRegressor(iterations=100, learning_rate=0.1, depth=6, verbose=0)
best_rf2 = RandomForestRegressor(n_estimators=100, max_depth=None, random_state=42)


# In[155]:


# Optuna Objective
def objective(trial):
    alpha = trial.suggest_float('alpha', 0.001, 10.0, log=True)
    final_estimator = Ridge(alpha=alpha)

    stack_model2 = StackingRegressor(
        estimators=[('xgb', best_xgb2), ('cat', best_cat2), ('rf', best_rf2)],
        final_estimator=final_estimator,
        passthrough=True,
        n_jobs=-1
    )

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    scores = cross_val_score(stack_model2, X_train, y_train, cv=kf, scoring=neg_rmsle_scorer, n_jobs=-1)
    return scores.mean()


# In[156]:


# Run study
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=5)


# In[157]:


import joblib

# Save the study
joblib.dump(study, 'optuna_STUDY.pkl')


# In[158]:


# Load the study later
study1 = joblib.load('optuna_STUDY.pkl')


# In[159]:


# Train best model
best_alpha = study1.best_params['alpha']

best_stacked_model_optuna = StackingRegressor(
    estimators=[
        ('xgb', best_xgb2),
        ('cat', best_cat2),
        ('rf', best_rf2)
    ],
    final_estimator=Ridge(alpha=best_alpha),
    passthrough=True,
    n_jobs=-1
)


# In[161]:


best_stacked_model_optuna.fit(X_train, y_train)


# In[162]:


joblib.dump(best_stacked_model_optuna, 'BEST_STACKED_MODEL_OPTUNA.pkl')


# In[163]:


best_stacked_model_optuna1 = joblib.load('BEST_STACKED_MODEL_OPTUNA.pkl')


# In[164]:


# Evaluate
stacked_preds = best_stacked_model_optuna1.predict(X_eval)
optuna_stacked_rmsle = np.sqrt(mean_squared_log_error(np.abs(y_eval), np.abs(stacked_preds)))
print("Best alpha:", best_alpha)
print("Stacked RMSLE:", stacked_rmsle)


# In[165]:


# Metrics
optuna_stacked_rmsle = np.sqrt(mean_squared_log_error(np.abs(y_eval), np.abs(stacked_preds)))
optuna_stacked_rmse = np.sqrt(mean_squared_error(y_eval, stacked_preds))
optuna_stacked_mse = mean_squared_error(y_eval, stacked_preds)
optuna_stacked_mae = mean_absolute_error(y_eval, stacked_preds)

# Results
optuna_results_stacked = pd.DataFrame({
    'Model': ['Stacked (XGB + CAT + RF → Linear)-optuna'],
    'RMSLE': [optuna_stacked_rmsle],
    'RMSE': [optuna_stacked_rmse],
    'MSE': [optuna_stacked_mse],
    'MAE': [optuna_stacked_mae]
}).round(3)

optuna_results_stacked


# # Compare performance of models

# In[166]:


# Combine all model results
all_results = [optuna_results_stacked, results_lr, results_arima, xgb_results, results_rf, results_gb, results_cat, results_stacked]
results_df = pd.concat(all_results, ignore_index=True)

# Sort by RMSLE, then RMSE, MSE, MAE
results_df = results_df.sort_values(by=['RMSLE', 'RMSE', 'MSE', 'MAE'], ascending=True).reset_index(drop=True)

# Truncate to 2 decimal places for display
def truncate(val, decimals=2):
    factor = 10 ** decimals
    return np.floor(val * factor) / factor

# Apply truncation for display
display_df = results_df.copy()
for col in ['RMSLE', 'RMSE', 'MSE', 'MAE']:
    display_df[col] = display_df[col].apply(lambda x: f"{truncate(x):.2f}")

# Get the best model
best_model = results_df.iloc[0]['Model']

# Print results
print("Sorted Results Based on RMSLE (and RMSE, MSE, MAE as tiebreakers):\n", display_df)
print("\nBest Model Based on RMSLE (with RMSE tiebreaker):", best_model)


# In[167]:


from tabulate import tabulate
print(tabulate(results_df, headers='keys', tablefmt='pretty', showindex=False))


# # Model for future predictions

# In[174]:


# Predict on evaluation data #X_eval #meta_test1
pred_values = best_stacked_model_optuna1.predict(X_eval)


# In[175]:


predictions_df = pd.DataFrame({
    'Actual': y_eval.values,
    'Predicted': pred_values
})
print(predictions_df.head())


# In[173]:


# Avoid divide-by-zero issues
epsilon = 1e-10

# Calculate percentage similarity for each prediction
percentage_similarity = 100 - (np.abs(y_eval - pred_values) / (np.abs(y_eval) + epsilon)) * 100

# Average similarity
average_similarity = np.mean(percentage_similarity)

print(f"Average Prediction Similarity: {average_similarity:.2f}%")


# If the prediction is perfect (i.e., y_pred ≈ y_true), the similarity is close to 100%

# In[176]:


import numpy as np

# Avoid divide-by-zero issues
epsilon = 1e-10

# Similarity calculation (higher is better, max 100%)
predictions_df['Similarity (%)'] = (
    (1 - (np.abs(predictions_df['Actual'] - predictions_df['Predicted']) / (np.abs(predictions_df['Actual']) + epsilon)))
    * 100
)

# Clip values between 0 and 100
predictions_df['Similarity (%)'] = predictions_df['Similarity (%)'].clip(lower=0, upper=100)

# Mean similarity
average_similarity = predictions_df['Similarity (%)'].mean()
print(f"\n🔍 Average Prediction Similarity: {average_similarity:.2f}%")


# In[177]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(12, 6))
sns.lineplot(data=predictions_df[['Actual', 'Predicted']])
plt.title('Actual vs Predicted Values')
plt.xlabel('Index')
plt.ylabel('Value')
plt.legend(['Actual', 'Predicted'])
plt.grid(True)
plt.tight_layout()
plt.show()


# In[178]:


plt.figure(figsize=(6, 6))
sns.scatterplot(x='Actual', y='Predicted', data=predictions_df)
plt.plot([predictions_df['Actual'].min(), predictions_df['Actual'].max()],
         [predictions_df['Actual'].min(), predictions_df['Actual'].max()],
         color='red', linestyle='--')
plt.title('Actual vs Predicted Scatter Plot')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[180]:


import matplotlib.pyplot as plt
import seaborn as sns

# Set figure size and style
plt.figure(figsize=(14, 6))
sns.set_style("whitegrid")

# Plot actual and predicted
plt.plot(predictions_df['Actual'].values, label='Actual', linewidth=2)
plt.plot(predictions_df['Predicted'].values, label='Predicted', linewidth=2)

# Graph details
plt.title('📈 Actual vs Predicted Line Plot', fontsize=16)
plt.xlabel('Index', fontsize=12)
plt.ylabel('Value', fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()


# In[181]:


plt.plot(predictions_df['Actual'].values[:100], label='Actual')
plt.plot(predictions_df['Predicted'].values[:100], label='Predicted')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


'''pip install flask'''


# In[ ]:


'''from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

"""# Load the saved model
with open('stacked_model.pkl', 'rb') as f:
    model = pickle.load(f)"""

@app.route('/')
def home():
    return "Stacked Model API is running!"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    input_data = pd.DataFrame([data])  # assuming data is a dict
    prediction = best_stacked_model1.predict(input_data)
    return jsonify({'prediction': prediction.tolist()})

if __name__ == '__main__':
    app.run(debug=True)
'''

