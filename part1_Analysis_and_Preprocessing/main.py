# Pandas library is used for handling tabular data
import pandas as pd

# NumPy is used for handling numerical series operations (addition, multiplication, and ...)

import numpy as np
# Sklearn library contains all the machine learning packages we need to digest and extract patterns from the data
from sklearn import linear_model, model_selection, metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

# Machine learning libraries used to build a decision tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree

# Sklearn's preprocessing library is used for processing and cleaning the data 
from sklearn import preprocessing

# for visualizing the tree
import pydotplus
from IPython.display import Image

df = pd.read_csv('store_sales.csv', encoding='latin1')
df.head()
df.info()

print(df.head(10))

# Display data types of each column
print(df.dtypes)
# Provide key statistical measures
statistics = df.describe()
print(statistics)

# Histograms for numerical columns
numerical_columns = ['Sales', 'Quantity', 'Discount', 'Profit']
df[numerical_columns].hist(bins=30, figsize=(10, 7))
plt.tight_layout()
plt.show()

# Box plots for numerical columns
plt.figure(figsize=(10, 7))
for i, col in enumerate(numerical_columns):
    plt.subplot(2, 2, i+1)
    sns.boxplot(y=df[col])
    plt.title(f'Box Plot of {col}')
plt.tight_layout()
plt.show()

# Convert data types
df['Order Date'] = pd.to_datetime(df['Order Date'])
df['Ship Date'] = pd.to_datetime(df['Ship Date'])
df['Postal Code'] = df['Postal Code'].astype(str)

# Aggregating data by order date
df_agg = df.groupby('Order Date').agg({'Sales': 'sum', 'Profit': 'sum'}).reset_index()

# Plotting the aggregated data
plt.figure(figsize=(12, 6))
plt.plot(df_agg['Order Date'], df_agg['Sales'], label='Sales')
plt.plot(df_agg['Order Date'], df_agg['Profit'], label='Profit')
plt.xlabel('Order Date')
plt.ylabel('Amount')
plt.title('Sales and Profit Over Time')
plt.legend()
plt.show()

# Checking for missing values
print(df.isnull().sum())

# Handling missing values (assuming dropping for this example)
df = df.dropna()

# Removing outliers using IQR method
Q1 = df[numerical_columns].quantile(0.25)
Q3 = df[numerical_columns].quantile(0.75)
IQR = Q3 - Q1

df = df[~((df[numerical_columns] < (Q1 - 1.5 * IQR)) | (df[numerical_columns] > (Q3 + 1.5 * IQR))).any(axis=1)]

# One-hot encode categorical columns
df = pd.get_dummies(df, columns=['Ship Mode', 'Segment', 'Country', 'City', 'State', 'Region', 'Category', 'Sub-Category', 'Product Name'])

# Selecting only numeric columns for correlation
numeric_df = df.select_dtypes(include=[np.number])

# Calculating correlation matrix
correlation_matrix = numeric_df.corr()

# Plotting the heatmap
plt.figure(figsize=(10, 7))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix')
plt.show()

# Summary
summary = """
1. Loaded the dataset and displayed the first ten instances.
2. Provided key statistical measures such as mean and standard deviation.
3. Visualized numerical columns using histograms and box plots.
4. Converted data types of columns to ensure all values are numerical.
5. Aggregated data by order date and visualized sales and profit over time.
6. Cleaned the data by handling missing values and removing outliers.
7. Identified correlated variables and visualized the correlation matrix.
"""
print(summary)







# # # Visualize numerical columns
# numerical_columns = df.select_dtypes(include=[np.number]).columns.tolist()

# # # Plot histograms for numerical columns
# for column in numerical_columns:
#     plt.figure(figsize=(10, 6))
#     plt.hist(df[column].dropna(), bins=30, edgecolor='k', alpha=0.7)
#     plt.title(f'Histogram of {column}')
#     plt.xlabel(column)
#     plt.ylabel('Frequency')
#     plt.show()

# # # Plot box plots for numerical columns
# for column in numerical_columns:
#     plt.figure(figsize=(10, 6))
#     sns.boxplot(data=df[column].dropna(), orient='h')
#     plt.title(f'Box plot of {column}')
#     plt.show()


# #     # Visualize numerical columns after conversion

# # # Correlation matrix
# plt.figure(figsize=(12, 8))
# correlation_matrix = df[numerical_columns].corr()
# sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
# plt.title('Correlation Matrix')
# plt.show()

# # # Aggregating data by a specific time period, for example, monthly
# if 'date_column_name' in df.columns:
#     df.set_index('date_column_name', inplace=True)
#     monthly_data = df.resample('M').mean()
    
#     # Plotting the aggregated data
#     plt.figure(figsize=(12, 8))
#     for column in numerical_columns:
#         plt.plot(monthly_data.index, monthly_data[column], label=column)
#     plt.title('Monthly Aggregated Data')
#     plt.xlabel('Date')
#     plt.ylabel('Values')
#     plt.legend()
#     plt.show()

# # # Visualizing non-numeric columns that were converted
# # # For example, if there was a categorical column that was converted to numeric
# categorical_columns = df.select_dtypes(include=['object']).columns.tolist()

# # # Bar plots for categorical columns
# for column in categorical_columns:
#     plt.figure(figsize=(10, 6))
#     sns.countplot(data=df, x=column)
#     plt.title(f'Bar plot of {column}')
#     plt.xlabel(column)
#     plt.ylabel('Count')
#     plt.show()

# # # Identifying trends, patterns, or anomalies
# # # Scatter plot matrix for numerical columns
# sns.pairplot(df[numerical_columns])
# plt.show()
