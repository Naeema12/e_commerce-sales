# E-Commerce Sales Analysis with Order Segmentation

## Project Overview
This project focuses on exploratory data analysis and order-level segmentation of an e-commerce sales dataset.
The goal is to analyze sales performance, product demand, country-based revenue distribution, and purchasing
patterns, and to demonstrate a complete data analytics workflow using Python.

In addition to exploratory analysis, a machine learning approach (K-Means clustering) is applied to segment
orders based on their value and size.

This project was created as part of my data analytics learning journey to practice real-world
EDA workflows.


## Dataset
- Type: E-commerce sales data
- Format: CSV
- Main columns:
  - OrderID
  - Date
  - Product
  - Category
  - Quantity
  - UnitPrice
  - Country
  - Total

 Note:  
The dataset does not include a customer identifier. Therefore, customer-level analysis is not possible.
Instead, order-level segmentation is performed, which is a realistic scenario in many transactional datasets.



## Tools & Technologies
- Python
- pandas
- matplotlib
- seaborn
- scikit-learn



## Exploratory Data Analysis (EDA)
The following steps were performed:
- Inspection of data structure using `info()`, `describe()`, `head()`, and `tail()`
- Frequency analysis of products and countries
- Creation of a `Total` column to calculate order revenue
- Analysis of:
  - Total and average order value
  - Revenue by country
  - Revenue by product
  - Average order value by category
- Time-based analysis of monthly sales trends


## Visualizations
The analysis includes multiple visualizations to support insights:
- Top 10 most sold products (bar chart)
- Top 10 countries by total revenue (pie chart)
- Monthly sales trend (line chart)
- Average order value by product category (bar chart)



## Machine Learning: Order Segmentation (K-Means)

### Problem Definition
Since customer identifiers are not available, K-Means clustering is applied at the **order level**
to identify different purchasing patterns.

### Feature Engineering
Orders were aggregated to create meaningful features:
- Total order value
- Total quantity per order
- Average unit price per order

### Model Preparation
- Features were standardized using `StandardScaler`
- The optimal number of clusters was selected using the **Elbow Method**

### Clustering & Interpretation
K-Means clustering was applied to segment orders into distinct groups:
- Small-value orders
- Medium-value orders
- High-value orders

The clusters were interpreted in business terms rather than focusing on model performance metrics.


## Key Insights
- A small number of orders contribute disproportionately to total revenue
- Revenue is concentrated in a limited number of countries
- Certain product categories have significantly higher average order values
- Order segmentation reveals clear differences between low-value and high-value purchasing behavior



## What I Learned
- How to perform exploratory data analysis on real-world transactional data
- How to create business-relevant metrics from raw data
- How to apply K-Means clustering responsibly when customer-level data is unavailable
- How to interpret machine learning results in a business context



