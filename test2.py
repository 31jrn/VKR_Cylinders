import pandas as pd
import numpy as np
from scipy.stats import kurtosis, zscore
from sklearn.impute import KNNImputer
from sklearn.cluster import DBSCAN

file_path = "VKR_dataset_test.xlsx"  # Имя файла Excel
df1 = pd.read_excel(file_path)

if df1.shape[1] < 2:
    raise ValueError("Ошибка: В файле недостаточно данных!")
df1.rename(columns={df1.columns[0]: "timestamp"}, inplace=True)  # Переименуем первый столбец
df1["timestamp"] = pd.to_datetime(df1["timestamp"], format="%d.%m.%Y %H:%M:%S", errors="coerce")
df1["Дата"] = df1["timestamp"].dt.date
df1["Время"] = df1["timestamp"].dt.time
df1 = df1.drop(columns=["timestamp"])  # Удаляем старую временную метку
columns_order = ["Дата", "Время"] + [col for col in df1.columns if col not in ["Дата", "Время"]]
df1 = df1[columns_order]
df1 = df1.dropna()
print("\n Данные после обработки временной метки:")
print(df1.head())
df1.to_csv("processed_data.csv", index=False)
print("\n Данные сохранены в 'processed_data.csv'")

df = pd.read_csv("processed_data.csv")
# === 2. Определяем нужные столбцы ===
columns_to_analyze = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Oil_temperature", "Motor_current"]
df_numeric = df[columns_to_analyze]


# === 3. Kurtosis Measure (Эксцесс) ===
def detect_outliers_kurtosis(df, threshold=3):
    kurt_values = df.apply(kurtosis)
    return kurt_values[abs(kurt_values) > threshold]


outliers_kurtosis = detect_outliers_kurtosis(df_numeric)
print("\n🔹 Выбросы по Kurtosis:\n", outliers_kurtosis)


# === 4. Z-score (Отклонение от среднего) ===
def detect_outliers_zscore(df, threshold=3):
    z_scores = np.abs(zscore(df))
    return np.where(z_scores > threshold)


outlier_rows, outlier_cols = detect_outliers_zscore(df_numeric)
print(f"\n🔹 Найдено {len(set(outlier_rows))} строк с выбросами по Z-score.")

# === 5. Очистка данных: KNNImputer ===
imputer = KNNImputer(n_neighbors=5)
df_knn_cleaned = pd.DataFrame(imputer.fit_transform(df_numeric), columns=df_numeric.columns)
print("\n✅ Данные очищены с помощью KNNImputer.")

# === 6. Очистка данных: DBSCAN ===
dbscan = DBSCAN(eps=1.5, min_samples=5)
labels = dbscan.fit_predict(df_knn_cleaned)
df_dbscan_cleaned = df_knn_cleaned[labels != -1]  # Удаляем шумовые точки
print("\n✅ Данные очищены с помощью DBSCAN.")

# === 7. Восстанавливаем "Дата" и "Время" ===
df_cleaned = df.iloc[df_dbscan_cleaned.index, :]

# === 8. Сохраняем обработанные данные ===
df_cleaned.to_csv("cleaned_data.csv", index=False)
print("\n✅ Финальные очищенные данные сохранены в 'cleaned_data.csv'.")


# === 9. Сглаживание данных методом скользящего среднего ===
def apply_moving_average(df, columns, window_size=3):
    smoothed_data = df.copy()
    for col in columns:
        smoothed_data[col] = df[col].rolling(window=window_size, center=True).mean()
    return smoothed_data


# Применяем сглаживание
df_smoothed = apply_moving_average(df_cleaned, columns_to_analyze, window_size=3)
df_smoothed = df_smoothed.dropna()  # Удаляем строки с NaN после сглаживания
df_smoothed.to_csv("smoothed_data.csv", index=False)
print("\n✅ Данные сглажены методом скользящего среднего и сохранены в 'smoothed_data.csv'.")

from sklearn.metrics import mean_squared_error


def calculate_metrics(original, smoothed, column):
    # Обрезаем оригинальные данные до длины сглаженных
    aligned_original = original.loc[smoothed.index]

    # Вычисляем MSE и RMSE
    mse = mean_squared_error(aligned_original[column], smoothed[column])  # MSE
    rmse = np.sqrt(mse)  # Корень из MSE для RMSE

    # Снижение дисперсии
    variance_reduction = (np.var(aligned_original[column]) - np.var(smoothed[column])) / np.var(
        aligned_original[column]) * 100

    print(f"RMSE для {column}: {rmse}")
    print(f"Снижение дисперсии для {column}: {variance_reduction:.2f}%")


# Убедимся, что NaN удалены перед расчетом метрик
df_smoothed = df_smoothed.dropna()  # Удаляем строки с NaN
calculate_metrics(df_cleaned, df_smoothed, "TP2")

"""
def calculate_metrics_with_nan(original, smoothed, column):
    # RMSE игнорирует NaN
    rmse = np.sqrt(np.nanmean((original[column] - smoothed[column]) ** 2))

    # Снижение дисперсии
    variance_reduction = (np.nanvar(original[column]) - np.nanvar(smoothed[column])) / np.nanvar(original[column]) * 100

    print(f"RMSE для {column} (с учетом NaN): {rmse}")
    print(f"Снижение дисперсии для {column} (с учетом NaN): {variance_reduction:.2f}%")


# Расчет метрик без обрезки данных
calculate_metrics_with_nan(df_cleaned, df_smoothed, "TP2")
"""

import matplotlib.pyplot as plt


def plot_comparison(original, smoothed, column):
    plt.figure(figsize=(12, 6))
    plt.plot(original[column], label="Original Data", alpha=0.7)
    plt.plot(smoothed[column], label="Smoothed Data (Moving Average)", alpha=0.9)
    plt.title(f"Сравнение данных - {column}")
    plt.xlabel("Index")
    plt.ylabel(column)
    plt.legend()
    plt.grid()
    plt.show()


# Сравнение для одного из параметров
plot_comparison(df_cleaned, df_smoothed, "TP2")
