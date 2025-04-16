import pandas as pd
import numpy as np
from typing import Tuple, List
import matplotlib.pyplot as plt


def estimate_period(x: np.ndarray, max_lag: int = 300) -> Tuple[int, List[float]]:
    """
    Вычисляет квазипериод T временного ряда по критерию минимума суммы квадратов ошибок.
    """
    errors = [(x[k:] - x[:-k]) ** 2 for k in range(2, max_lag)]
    S = [np.sum(err) for err in errors]
    T = np.argmin(S) + 2  # +2, т.к. индексы начинались с k=2
    return T, S


def load_and_prepare_timeseries(
        path: str,
        date_col: str = "Date",
        time_col: str = "Time",
        drop_cols: List[str] = None,
        max_period_lag: int = 300,
        period_feature_index: int = 0,
) -> Tuple[np.ndarray, pd.Series, int]:
    """
    Загрузка и подготовка временного ряда из CSV:
    - объединяет столбцы даты и времени;
    - удаляет ненужные столбцы;
    - возвращает матрицу признаков, временной индекс и период.
    """
    df = pd.read_csv(path)

    # Объединить дату и время
    df["timestamp"] = pd.to_datetime(df[date_col] + " " + df[time_col])
    df = df.sort_values(by="timestamp").reset_index(drop=True)

    # Удалить лишние столбцы
    base_cols = [date_col, time_col, "timestamp"]
    if drop_cols:
        base_cols.extend(drop_cols)
    feature_df = df.drop(columns=base_cols)

    # Преобразовать в numpy
    X = feature_df.to_numpy().T  # (n_features, n_timesteps)

    # Определение периода по одному признаку (по умолчанию X[0])
    signal = X[period_feature_index]
    period, errors = estimate_period(signal, max_period_lag)

    # График ошибок по разным сдвигам
    plt.figure(figsize=(8, 4))
    plt.plot(range(2, len(errors) + 2), errors)
    plt.title(f"Оценка периода: минимальная ошибка при T={period}")
    plt.xlabel("T")
    plt.ylabel("S(T)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return X, df["timestamp"], period


X, timestamps, T = load_and_prepare_timeseries("smoothed_data.csv")
print(f"Размерность массива X: {X.shape}")
print(f"Оцененный период: {T}")
