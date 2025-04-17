# === 1. Импорт библиотек ===
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, zscore
from sklearn.cluster import DBSCAN
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# === 2. Загрузка и подготовка данных ===
file_path = "VKR_dataset_test.xlsx"
df1 = pd.read_excel(file_path)
df1.rename(columns={df1.columns[0]: "timestamp"}, inplace=True)
df1["timestamp"] = pd.to_datetime(df1["timestamp"], format="%d.%m.%Y %H:%M:%S", errors="coerce")
df1["Дата"] = df1["timestamp"].dt.date
df1["Время"] = df1["timestamp"].dt.time
df1 = df1.drop(columns=["timestamp"])
columns_order = ["Дата", "Время"] + [col for col in df1.columns if col not in ["Дата", "Время"]]
df1 = df1[columns_order].dropna()
df1.to_csv("processed_data.csv", index=False)
print("\n✅ Данные сохранены в 'processed_data.csv'")

# === 3. Выбор переменных для анализа ===
df = pd.read_csv("processed_data.csv")
columns_to_analyze = [col for col in df.columns if col not in ["Дата", "Время"]]


# === 4. Удаление выбросов по Z-score ===
def remove_outliers_zscore(df, threshold=3):
    z_scores = np.abs(zscore(df))
    mask = (z_scores < threshold).all(axis=1)
    return df[mask]


df_cleaned = remove_outliers_zscore(df[columns_to_analyze])
print("\n✅ Данные очищены от выбросов по Z-score")

# === 5. Очистка данных с помощью DBSCAN ===
dbscan = DBSCAN(eps=1.5, min_samples=5)
labels = dbscan.fit_predict(df_cleaned)
df_dbscan_cleaned = df_cleaned[labels != -1]
print("\n✅ Шум удалён методом DBSCAN")

# === 6. Восстановление "Дата" и "Время" ===
df_cleaned_full = df.loc[df_dbscan_cleaned.index, :]
df_cleaned_full.to_csv("cleaned_data.csv", index=False)
print("\n✅ Финальные очищенные данные сохранены в 'cleaned_data.csv'")


# === 7. Сглаживание ===
def apply_moving_average(df, columns, window_size=3):
    smoothed_data = df.copy()
    for col in columns:
        smoothed_data[col] = df[col].rolling(window=window_size, center=True).mean()
    return smoothed_data.dropna()


df_smoothed = apply_moving_average(df_cleaned_full, columns_to_analyze)
df_smoothed.to_csv("smoothed_data.csv", index=False)
print("\n✅ Данные сглажены и сохранены в 'smoothed_data.csv'")


# === 8. Оценка сглаживания ===
def calculate_metrics(original, smoothed, column):
    aligned_original = original.loc[smoothed.index]
    rmse = np.sqrt(mean_squared_error(aligned_original[column], smoothed[column]))
    variance_reduction = (np.var(aligned_original[column]) - np.var(smoothed[column])) / np.var(
        aligned_original[column]) * 100
    print(f"\n🔍 {column}: RMSE = {rmse:.4f}, Снижение дисперсии = {variance_reduction:.2f}%")


for col in columns_to_analyze:
    calculate_metrics(df_cleaned_full, df_smoothed, col)


# === 9. Структура цилиндрической модели ===
# (1) Определение периода
# (2) Развёртка в цилиндрическую модель
# (3) Метод наименьших квадратов (МНК)
# (4) Псевдоградиентное обновление параметров

def estimate_period(series, min_period=2, max_period=100):
    errors = []
    for p in range(min_period, max_period + 1):
        segments = series[:len(series) // p * p].reshape(-1, p)
        mean_segment = segments.mean(axis=0)
        error = np.mean((segments - mean_segment) ** 2)
        errors.append((p, error))
    best_period, _ = min(errors, key=lambda x: x[1])
    return best_period


def build_cylindrical_model(series, period):
    n = len(series)
    num_cycles = n // period
    return series[:num_cycles * period].reshape(num_cycles, period)


def least_squares_fit(cyl_model):
    X = cyl_model[:-1]
    Y = cyl_model[1:]
    params = np.linalg.pinv(X) @ Y
    return params


def pseudo_gradient_update(X, Y, alpha=0.001, iterations=100):
    params = np.random.randn(X.shape[1], X.shape[1])
    for _ in range(iterations):
        grad = -2 * X.T @ (Y - X @ params) / len(X)
        params -= alpha * grad
    return params


for col in columns_to_analyze:
    print(f"🔄 Обработка признака: {col}")
    series = df_smoothed[col].values
    period = estimate_period(series)
    print(f"📏 Определённый период для {col}: {period}")
    cyl_model = build_cylindrical_model(series, period)
    params_ls = least_squares_fit(cyl_model)
    params_pg = pseudo_gradient_update(cyl_model[:-1], cyl_model[1:])
    print(f"✅ Параметры признака {col} обучены методом МНК и псевдоградиентным методом")
print("\n✅ Параметры обучены методом МНК и псевдоградиентным методом")


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
plot_comparison(df_cleaned, df_smoothed, "Motor_current")
