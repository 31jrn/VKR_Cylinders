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