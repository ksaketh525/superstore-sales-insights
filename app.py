import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# --- Load Dataset ---
df = pd.read_excel("Sample - Superstore.xls")  # changed from read_csv to read_excel

# --- Data Cleaning & Feature Engineering ---
df['Order Date'] = pd.to_datetime(df['Order Date'], errors='coerce')
df['Year'] = df['Order Date'].dt.year
df['Month'] = df['Order Date'].dt.month
df['Profit_Margin'] = df['Profit'] / df['Sales']

# --- App Title ---
st.title("📊 Superstore Sales Insights Dashboard")

# --- Sidebar Filters ---
category_filter = st.sidebar.multiselect(
    "Select Product Category",
    options=df['Category'].dropna().unique(),
    default=df['Category'].dropna().unique()
)

region_filter = st.sidebar.multiselect(
    "Select Region",
    options=df['Region'].dropna().unique(),
    default=df['Region'].dropna().unique()
)

filtered_df = df[
    (df['Category'].isin(category_filter)) &
    (df['Region'].isin(region_filter))
]

# --- Monthly Sales Trend ---
st.subheader("📈 Monthly Sales Trend")
monthly_sales = (
    filtered_df.groupby(filtered_df['Order Date'].dt.to_period('M'))['Sales']
    .sum()
    .reset_index()
)
monthly_sales['Order Date'] = monthly_sales['Order Date'].dt.to_timestamp()

fig, ax = plt.subplots(figsize=(10,5))
ax.plot(monthly_sales['Order Date'], monthly_sales['Sales'], marker='o', color='royalblue')
ax.set_xlabel("Month")
ax.set_ylabel("Sales ($)")
ax.set_title("Monthly Sales Trend")
plt.xticks(rotation=45)
st.pyplot(fig)

# --- Top 10 Products ---
st.subheader("🏆 Top 10 Products by Sales")
top_products = (
    filtered_df.groupby('Product Name')['Sales']
    .sum()
    .sort_values(ascending=False)
    .head(10)
)
st.bar_chart(top_products)

# --- Regional Sales & Profit Table ---
st.subheader("🌍 Regional Sales & Profit Summary")
region_summary = (
    filtered_df.groupby('Region')[['Sales', 'Profit']]
    .sum()
    .reset_index()
)
st.dataframe(region_summary.style.format({"Sales": "${:,.2f}", "Profit": "${:,.2f}"}))

# --- Profit vs Sales Scatter Plot ---
st.subheader("💰 Profit vs Sales by Category")
fig2, ax2 = plt.subplots(figsize=(8,5))
sns.scatterplot(data=filtered_df, x='Sales', y='Profit', hue='Category', ax=ax2)
ax2.set_title("Profit vs Sales by Category")
st.pyplot(fig2)

# --- Simple Forecast: Linear Regression ---
st.subheader("📅 Sales Forecast: Next 6 Months")
monthly_sales['Month_Index'] = range(len(monthly_sales))
X = monthly_sales[['Month_Index']]
y = monthly_sales['Sales']

model = LinearRegression()
model.fit(X, y)

future_index = pd.DataFrame({'Month_Index': range(len(monthly_sales), len(monthly_sales) + 6)})
future_sales = model.predict(future_index)

forecast_df = pd.DataFrame({
    "Month Ahead": [f"Month {i+1}" for i in range(6)],
    "Predicted Sales ($)": future_sales.round(2)
})

st.table(forecast_df)

# --- Summary KPIs ---
st.subheader("📌 Key Metrics")
col1, col2 = st.columns(2)
col1.metric("Total Sales", f"${filtered_df['Sales'].sum():,.2f}")
col2.metric("Total Profit", f"${filtered_df['Profit'].sum():,.2f}")
