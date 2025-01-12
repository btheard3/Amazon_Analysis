#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# Streamlit Title and Description
st.title("Amazon Sales Analysis")
st.markdown("""
This app analyzes Amazon sales data and compares machine learning models for sales prediction.
""")

# Load and Inspect the Dataset
data_path = "Amazon Sale Report.csv"

try:
    # Load the dataset
    st.write("### Loading Dataset...")
    data = pd.read_csv(data_path, low_memory=False)
    st.write("### Data Preview")
    st.write(data.head())
    
    # Initial Inspection
    st.write("### Dataset Information")
    st.write(f"Shape: {data.shape}")
    st.write(data.info())

    # Check for missing values
    st.write("### Missing Values")
    missing_values = data.isnull().sum()
    st.write(missing_values)

    # Data Cleaning
    st.write("### Data Cleaning")
    data = data.drop(['Unnamed: 22', 'promotion-ids'], axis=1, errors='ignore')
    data['Courier Status'] = data['Courier Status'].fillna('Unknown')
    data['currency'] = data['currency'].fillna(data['currency'].mode()[0])
    data = data.dropna(subset=['Amount', 'ship-city', 'ship-state', 'ship-postal-code', 'ship-country'])
    data['Date'] = pd.to_datetime(data['Date'], format='%m-%d-%y', errors='coerce')
    data = data.drop(['fulfilled-by'], axis=1, errors='ignore')
    st.write("Cleaned Data Preview:")
    st.write(data.head())

    # Feature Engineering
    st.write("### Feature Engineering")
    data['Month'] = data['Date'].dt.month
    data['Year'] = data['Date'].dt.year
    st.write(data[['Date', 'Month', 'Year']].head())

    # Exploratory Data Analysis
    st.write("### Exploratory Data Analysis")

    # Total Sales per Category
    st.write("#### Total Sales per Category")
    category_sales = data.groupby('Category')['Amount'].sum().sort_values(ascending=False)
    st.bar_chart(category_sales)

    # Monthly Sales Trend
    st.write("#### Monthly Sales Trend")
    monthly_sales = data.groupby(['Year', 'Month'])['Amount'].sum().reset_index()
    monthly_sales['Date'] = pd.to_datetime(monthly_sales[['Year', 'Month']].assign(Day=1))
    st.line_chart(monthly_sales.set_index('Date')['Amount'])

    # Correlation Heatmap
    st.write("#### Correlation Heatmap")
    numeric_features = ['Qty', 'Amount']
    corr_matrix = data[numeric_features].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    st.pyplot(fig)

    # Dimensionality Reduction (PCA)
    st.write("### Dimensionality Reduction (PCA)")
    data_numeric = data[numeric_features].dropna()
    pca = PCA(n_components=2)
    principal_components = pca.fit_transform(data_numeric)
    data['PC1'], data['PC2'] = principal_components[:, 0], principal_components[:, 1]
    st.write(data[['PC1', 'PC2']].head())

    # Train-Test Split
    st.write("### Train-Test Split")
    X = data[['Qty', 'PC1', 'PC2', 'Month', 'Year']]
    y = data['Amount']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    st.write(f"Training Set: {X_train.shape}, Testing Set: {X_test.shape}")

    # Model Training and Evaluation
    st.write("### Model Training and Evaluation")

    # Decision Tree
    dt = DecisionTreeRegressor(random_state=42)
    dt.fit(X_train, y_train)
    dt_rmse = np.sqrt(mean_squared_error(y_test, dt.predict(X_test)))
    st.write(f"Decision Tree RMSE: {dt_rmse}")

    # Random Forest
    rf = RandomForestRegressor(random_state=42)
    rf.fit(X_train, y_train)
    rf_rmse = np.sqrt(mean_squared_error(y_test, rf.predict(X_test)))
    st.write(f"Random Forest RMSE: {rf_rmse}")

    # K-Nearest Neighbors
    knn = KNeighborsRegressor()
    knn.fit(X_train, y_train)
    knn_rmse = np.sqrt(mean_squared_error(y_test, knn.predict(X_test)))
    st.write(f"KNN RMSE: {knn_rmse}")

    # Model Comparison
    st.write("### Model Comparison")
    model_results = pd.DataFrame({
        "Model": ["Decision Tree", "Random Forest", "KNN"],
        "RMSE": [dt_rmse, rf_rmse, knn_rmse]
    })
    st.bar_chart(model_results.set_index("Model")["RMSE"])

except FileNotFoundError:
    st.error("The file 'Amazon_Sales_Report.csv' was not found. Ensure it exists in the repository.")
except Exception as e:
    st.error(f"An error occurred: {e}")
