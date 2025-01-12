# Amazon Sales Analysis

This Streamlit app provides insights into Amazon sales data through various visualizations and machine learning models. It helps explore patterns, trends, and correlations within the dataset and predicts sales using advanced regression techniques.

## 🔗 Live Demo
[Click here to explore the app!](https://amazonanalysis.streamlit.app/)

## Key Features
1. **Data Cleaning**: Handles missing values, removes irrelevant columns, and standardizes date formats for analysis.
2. **Feature Engineering**: Extracts temporal features (Month, Year) and applies dimensionality reduction using PCA for improved modeling.
3. **Exploratory Data Analysis (EDA)**: Provides interactive visualizations to explore sales trends, category contributions, and correlations.
4. **Machine Learning Models**: Implements Decision Tree, Random Forest, and K-Nearest Neighbors regressors to predict sales, with an RMSE comparison.

---

## Visualizations

### 1. **Total Sales per Category**
   - A bar chart displays total sales revenue for each product category.
   - **Insight**: Identify top-performing categories and focus on their growth.
   - ![image](https://github.com/user-attachments/assets/f20c945e-177a-4f65-97c7-324f782c63d5)


### 2. **Monthly Sales Trend**
   - A line chart shows how sales evolve monthly over multiple years.
   - **Insight**: Detect seasonal patterns or periods of high/low sales.
   - ![image](https://github.com/user-attachments/assets/3fb49cd8-25c8-48ec-a496-d224f3da11a4)


### 3. **Correlation Heatmap**
   - A heatmap highlights relationships between numerical features (e.g., sales amount and quantity).
   - **Insight**: Determine which features strongly influence sales performance.
   - ![image](https://github.com/user-attachments/assets/4c309a0c-5a9c-4256-94de-3bb3cb8409be)


### 4. **Year-over-Year Sales Comparison**
   - A multi-line chart compares yearly sales trends across months.
   - **Insight**: Track growth or decline in sales over time.
   - ![image](https://github.com/user-attachments/assets/379d7443-f305-4294-a6c5-91faf9e0d07f)


### 5. **Category-wise Sales Contribution**
   - A pie chart illustrates the percentage contribution of each category to overall sales.
   - **Insight**: Allocate resources based on category importance.
   - ![image](https://github.com/user-attachments/assets/a8c0e22d-0584-4fc6-890a-82a253077121)

---

## Machine Learning Models
- **Decision Tree Regressor**: Simple and interpretable but prone to overfitting.
- **Random Forest Regressor**: More robust, handles non-linear data well.
- **K-Nearest Neighbors Regressor**: Effective for simpler datasets with fewer dimensions.
- **Comparison**: RMSE values are calculated for each model to identify the best performer.

---

## Business Context
Amazon generates significant sales data, which can be a goldmine for understanding customer behavior, optimizing inventory, and boosting profitability. This app enables businesses to:
- **Identify Trends**: Detect seasonal and category-specific trends for better decision-making.
- **Optimize Resources**: Allocate inventory, marketing, and workforce efficiently.
- **Predict Sales**: Use machine learning models to forecast sales and prepare for future demand.
- **Customer Insights**: Identify high-value customers and tailor strategies to maximize revenue.

By leveraging the insights provided by this app, businesses can stay competitive, improve customer satisfaction, and drive sustainable growth.
