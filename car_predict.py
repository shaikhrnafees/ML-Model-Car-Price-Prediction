# -*- coding: utf-8 -*-
"""Car_predict.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1tLKV_OV0gXwh72f5fZ2hq0YN6FzWrat5

# **STEP 1: Problem Statement**
The task involves predicting car selling prices using the "cardekho" dataset, which includes various features such as name, year, km_driven, fuel, seller_type, transmission, owner, mileage, engine, max_power, and seats. The problem statement is to build a machine learning model that accurately predicts the selling prices based on these car features and histories.

The initial steps include cleaning the dataset by handling missing values, correcting data types, and addressing outliers to ensure data quality. Exploratory Data Analysis (EDA) will be conducted to understand the relationships between the features and the target variable, selling_price, using visualizations to uncover patterns and correlations.

Data preprocessing will involve encoding categorical variables (e.g., fuel, seller_type, transmission, owner) and scaling numerical features (e.g., km_driven, mileage, engine, max_power). After preprocessing, a machine learning model, such as RandomForestRegressor, will be created to predict selling prices. Hyperparameter tuning will be performed to optimize the model’s performance, ensuring accurate and reliable predictions based on the given car features and histories.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

"""# **Step 2: Data Collection**


"""

df=pd.read_csv("/content/cardekho.csv")

df.head()

# Extract the first part from the 'name' column because we need only the brand name
df['name'] = df['name'].apply(lambda x: x.split()[0])

df.head(5)

df.info()  # as you can see there are object columns according to dataframe

df.shape # rows,columns

df.describe() # some important information(mean,std, count , outliers view)

df.isna().sum() # null values lets check the weightage is higher 50 % or not

(df.isna().sum()/len(df))*100  # less than 50% so we can not drop

# Heatmap for showing missing values in data
plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

"""# **Step 3: Data Preprocessing - Part 1**"""

from sklearn.impute import SimpleImputer

# filling the missing values with mode because i cant take other method to justify this most frequent is suitable
s=SimpleImputer(strategy='most_frequent')
df[['mileage(km/ltr/kg)', 'engine', 'max_power', 'seats']] = s.fit_transform(df[['mileage(km/ltr/kg)', 'engine', 'max_power', 'seats']])

df.isna().sum() # now no missing values

df.duplicated().sum()

df.drop_duplicates(keep='first',inplace=True)

df.duplicated().sum() # no more duplicates

df.shape # rows,columns

# Now checking how much the data we cleaned
data_cleaned=round(((8128-6907)/8128)*100,1)
print(f"Cleaned Data = {data_cleaned} %")

df.head()

"""# **Step 4: Exploratory Data Analysis (EDA)**"""

# Distributor counts in car variants

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='fuel')
plt.title('Distribution of Fuel Types')
plt.xlabel('Fuel Type')
plt.ylabel('Count')
plt.show()

# Distribution of Seller Types

plt.figure(figsize=(8, 8))
df['seller_type'].value_counts().plot.pie(autopct='%1.1f%%', startangle=90)
plt.title('Distribution of Seller Types')
plt.ylabel('')
plt.show()

# Transmission Types by Fuel

plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='fuel', hue='transmission')
plt.title('Transmission Types by Fuel')
plt.xlabel('Fuel Type')
plt.ylabel('Count')
plt.legend(title='Transmission')
plt.show()

# Selling Price vs. Mileage

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='mileage(km/ltr/kg)', y='selling_price')
plt.title('Selling Price vs. Mileage')
plt.xlabel('Mileage (km/ltr/kg)')
plt.ylabel('Selling Price')
plt.show()

# Selling Price by Transmission Type
# There are outliers here but according to data we con not remove them because some car might be high price some care has low price.
# Example = BMW Car >>> Maruti Cars
plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='transmission', y='selling_price')
plt.title('Selling Price by Transmission Type')
plt.xlabel('Transmission Type')
plt.ylabel('Selling Price')
plt.show()

"""# **Step 3: Data Preprocessing- Part 2**"""

df.nunique() # unique values in each columns

df["owner"].unique()

# Map Encoding to justify the value

custom_encoding = {
    'First Owner': 1,
    'Second Owner': 2,
    'Third Owner': 3,
    'Fourth & Above Owner': 4,
    'Test Drive Car': 5
}

# Replace values in 'owner' column using custom encoding
df['owner'] = df['owner'].map(custom_encoding)

df["fuel"].unique()

custom_encoding_fuel = {
    'Diesel': 1,
    'Petrol': 2,
    'LPG': 3,
    'CNG' : 4,
}

df['fuel'] = df['fuel'].map(custom_encoding_fuel)

df["seller_type"].unique()

custom_encoding_seller = {
    'Individual': 1,
    'Dealer': 2,
    'Trustmark Dealer': 3
}
df['seller_type'] = df['seller_type'].map(custom_encoding_seller)

df['transmission'].unique()

custom_encoding_transmission = {
    'Manual': 2,
    'Automatic': 1,
}
df['transmission'] = df['transmission'].map(custom_encoding_transmission)

df['name'].unique()

from sklearn.preprocessing import LabelEncoder # Now doing the lable coding for higher unique values for this Lable Encoder is perfect fit.

label_encoder = LabelEncoder()

df['name'] = label_encoder.fit_transform(df['name'])

label_encoder = LabelEncoder()

df['mileage(km/ltr/kg)'] = label_encoder.fit_transform(df['mileage(km/ltr/kg)'])

label_encoder = LabelEncoder()

df['engine'] = label_encoder.fit_transform(df['engine'])

label_encoder = LabelEncoder()

df['max_power'] = label_encoder.fit_transform(df['max_power'])

label_encoder = LabelEncoder()

df['seats'] = label_encoder.fit_transform(df['seats'])

df.head(2) # Labeling Completed

df.info() # before and after info now we have all int datatype

# Checking relations

plt.figure(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt='.2f', annot_kws={'size': 10})
plt.title('Correlation Heatmap')
plt.show()

"""# **Step 5: Model Selection, Training & Evaluation**"""

# Deviding data to dependent and independent variable

input_data=df.drop('selling_price',axis=1)
output_data=df['selling_price']

from sklearn.preprocessing import StandardScaler

# Scaling the data to handle imbalance data.

ss=StandardScaler()
input_data=ss.fit_transform(input_data)

from sklearn.model_selection import train_test_split

# diving data inton train and test both depnedent and independent with ration 80-20

x_train,x_test,y_train,y_test=train_test_split(input_data,output_data,test_size=0.2,random_state=42)

from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# Created the model on linear regression and check accuracy whic is very poor

lr=LinearRegression()
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
r2_score(y_test,y_pred)*100

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

# In random forest getting a good accuracy

rf = RandomForestRegressor(n_estimators = 100)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
r2_score(y_test, y_pred)*100

# Plotting actual vs predicted values with different colors

plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred, color='blue', alpha=0.6, label='Predicted')
sns.scatterplot(x=y_test, y=y_test, color='red', alpha=0.4, label='Actual')

plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Actual vs Predicted Values')
plt.legend()
plt.grid(True)
plt.show()

# Checking the accuracyt of the model by extracting the sample data from same data set

x=df.drop('selling_price',axis=1)

sample_index = 0  # Change this index to select a different row from the DataFrame
sample_data = x.iloc[sample_index:sample_index+1]

# Apply the same scaling to the sample data
sample_data_scaled = scaler.transform(sample_data)

# Predict using the trained model
predicted_price = rf.predict(sample_data_scaled)
print("Predicted Price for sample data at index", sample_index, ":", predicted_price)

df.head(2)

# AS we can see here the price diffrence in index 1 for the accuracy

"""# **Thank you for your time. I look forward to your response.**

**Nafees Shaikh**
"""