import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "6"  # Укажи нужное количество ядер

# === 1. Загружаем данные ===
file_path = "cleaned_data.csv"  # Файл после обработки выбросов
df = pd.read_csv(file_path)

# === 2. Определяем нужные столбцы ===
columns_to_analyze = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]
df_numeric = df[columns_to_analyze]

# === 3. Масштабируем данные (Standard Scaling) ===
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_numeric)

# === 4. K-Means Clustering ===
k = 4  # Количество кластеров (можно менять)
kmeans = KMeans(n_clusters=k, random_state=42)
clusters = kmeans.fit_predict(df_scaled)

df["Cluster_KMeans"] = clusters

# Визуализируем K-Means кластеры
plt.scatter(df_scaled[:, 0], df_scaled[:, 1], c=clusters, cmap='viridis', alpha=0.5)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=200, c='red', marker='X', label="Centroids")
plt.title("K-Means Clustering")
plt.xlabel(columns_to_analyze[0])
plt.ylabel(columns_to_analyze[1])
plt.legend()
plt.figure()

# === 5. PCA (Метод главных компонент) ===
pca = PCA(n_components=2)  # Оставляем 2 главные компоненты
df_pca = pca.fit_transform(df_scaled)

plt.scatter(df_pca[:, 0], df_pca[:, 1], c=clusters, cmap='plasma', alpha=0.5)
plt.title("PCA Visualization")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

df["PCA_1"] = df_pca[:, 0]
df["PCA_2"] = df_pca[:, 1]
plt.figure()

# === 6. Робастное масштабирование (Robust Scaling) ===
robust_scaler = RobustScaler()
df_robust_scaled = robust_scaler.fit_transform(df_numeric)

# === 7. DBSCAN Clustering ===
dbscan = DBSCAN(eps=4, min_samples=38)  # eps и min_samples можно подбирать
clusters_dbscan = dbscan.fit_predict(df_robust_scaled)

df["Cluster_DBSCAN"] = clusters_dbscan

# Визуализируем DBSCAN
plt.scatter(df_robust_scaled[:, 0], df_robust_scaled[:, 1], c=clusters_dbscan, cmap='rainbow', alpha=0.5)
plt.title("DBSCAN Clustering")
plt.xlabel(columns_to_analyze[0])
plt.ylabel(columns_to_analyze[1])
# Уникальные метки кластеров
unique_labels = set(dbscan.labels_)

# Создаем кастомную легенду
handles = [
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=plt.cm.plasma(i / len(unique_labels)), markersize=10,
               label=f'Cluster {i}') for i in unique_labels]
plt.legend(handles=handles, title="Clusters")
plt.show()

# === 8. Сохраняем результат ===
df.to_csv("processed_with_clusters.csv", index=False)
print("\n✅ Данные сохранены в 'processed_with_clusters.csv'.")


# === 9. Сглаживание данных методом скользящего среднего ===
def apply_moving_average(df, columns, window_size=5):
    smoothed_data = df.copy()
    for col in columns:
        smoothed_data[col] = df[col].rolling(window=window_size, center=True).mean()
    return smoothed_data


# Применяем сглаживание
df_smoothed = apply_moving_average(df_cleaned, columns_to_analyze, window_size=5)
df_smoothed = df_smoothed.dropna()  # Удаляем строки с NaN после сглаживания
df_smoothed.to_csv("smoothed_data.csv", index=False)
print("\n✅ Данные сглажены методом скользящего среднего и сохранены в 'smoothed_data.csv'.")
