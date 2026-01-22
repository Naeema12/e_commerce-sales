
"""ecomerce sales analysis"""

import pandas as pd

df = pd.read_csv("ecommerce_sales (1).csv")

df.info()
df.describe()
df.head()
df.tail()

bu kısımda explarotary data analysis yapıyoruz veri setimizi keşfediyoruz

df["Product"].value_counts().head(10)  # hangi urun kaç kez satıldığını gosterir
df["Country"].value_counts().head(10)  # ulkere gore siparış sayısını gosterir
df["Total"] = df["Quantity"] * df["UnitPrice"] # her siparişin toplam tutarini hesaplar
df[["Product", "Quantity", "UnitPrice", "Total"]].head()
df["Total"].sum() # tum siparişlerin toplam değeri
df["Total"].mean() # her siparişin ortalama tutarını hesaplar

df.groupby("Country")["Total"].sum().sort_values(ascending=False).head(10) #ülkelere göre toplam gelir
df.groupby("Product")["Total"].sum().sort_values(ascending=False).head(10) # en çok gelir getiren ürünler
df.groupby("Category")["Total"].mean().sort_values(ascending=False) # her katagori için ortalama şipariş geliri
df["Date"] = pd.to_datetime(df["Date"])  # tarih sütünü tarih formatına çevire
df.groupby(df["Date"].dt.to_period("M"))["Total"].sum()

import matplotlib.pyplot as plt
import seaborn as sns

# en çok satılan urunler
top_products = df["Product"].value_counts().head(10)

plt.figure(figsize=(8,4))
sns.barplot(x=top_products.values, y=top_products.index)
plt.title("En Çok Satılan 10 Ürün")
plt.xlabel("Satış Sayısı")
plt.ylabel("Ürün Adı")
plt.tight_layout()
plt.show()

# ulkelere gore toplam gelir
country_sales = df.groupby("Country")["Total"].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(6,6))
plt.pie(country_sales.values, labels=country_sales.index, autopct="%1.1f%%", startangle=140)
plt.title("Ülkelere Göre Toplam Gelir (En Yüksek 10 Ülke)")
plt.axis("equal")  # Daireyi düzgün gösterir
plt.show()

# aylara göre guruplandırma
monthly_sales = df.groupby(df["Date"].dt.to_period("M"))["Total"].sum()

plt.figure(figsize=(8,4))
monthly_sales.plot(kind="line", marker="o")
plt.title("Aylık Satış Trendi")
plt.xlabel("Ay")
plt.ylabel("Toplam Gelir")
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()
plt.show()

# katagorileri göre ortalama sipariş geliri
category_avg = df.groupby("Category")["Total"].mean().sort_values(ascending=False)

plt.figure(figsize=(8,4))
sns.barplot(x=category_avg.values, y=category_avg.index)
plt.title("Kategorilere Göre Ortalama Sipariş Geliri")
plt.xlabel("Ortalama Gelir")
plt.ylabel("Kategori")
plt.tight_layout()
plt.show()

order_df = df.groupby("OrderID").agg(
    total_spent=("Total", "sum"),
    total_quantity=("Quantity", "sum"),
    avg_unit_price=("UnitPrice", "mean")
).reset_index()

order_df.head()

from sklearn.preprocessing import StandardScaler

features = ["total_spent", "total_quantity", "avg_unit_price"]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(order_df[features])

from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

inertia = []

for k in range(2, 8):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)

plt.plot(range(2, 8), inertia, marker="o")
plt.title("Elbow Method for Order Segmentation")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()

kmeans = KMeans(n_clusters=3, random_state=42)
order_df["Cluster"] = kmeans.fit_predict(scaled_features)

order_df.head()

cluster_summary = order_df.groupby("Cluster").agg(
    avg_spent=("total_spent", "mean"),
    avg_quantity=("total_quantity", "mean"),
    avg_price=("avg_unit_price", "mean"),
    orders=("OrderID", "count")
).reset_index()

cluster_summary

import seaborn as sns

plt.figure(figsize=(6,4))
sns.scatterplot(
    data=order_df,
    x="total_quantity",
    y="total_spent",
    hue="Cluster",
    palette="Set2"
)
plt.title("Order Segmentation by Quantity and Spend")
plt.show()
