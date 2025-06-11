import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.cluster import DBSCAN
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


# ——————————————
#  ПОДГОТОВКА И ДЕКОРАТОРЫ (без изменений)
# ——————————————

def load_data(file_path):
    df = pd.read_csv(file_path, sep=';')
    return df


def menu(columns):
    print("\nВыберите характеристику для анализа:")
    for idx, col in enumerate(columns):
        print(f"{idx}: {col}")
    print(f"{len(columns)}: Выход")
    while True:
        choice = input("Введите номер характеристики: ")
        if choice.isdigit():
            idx = int(choice)
            if idx == len(columns):
                print("Выход.")
                sys.exit()
            elif 0 <= idx < len(columns):
                print(f"Выбран столбец: {columns[idx]}\n")
                return columns[idx]
        print("Неверный ввод, попробуйте снова.")


def full_preprocessing(df, column, window=3, z_thresh=3.0):
    df_clean = df.dropna(subset=[column]).copy()
    df_clean = df_clean[np.abs(zscore(df_clean[column])) < z_thresh]
    X = np.hstack([np.arange(len(df_clean)).reshape(-1, 1),
                   df_clean[column].values.reshape(-1, 1)])
    labels = DBSCAN(eps=20, min_samples=5).fit_predict(X)
    df_clean['cluster'] = labels
    df_clean = df_clean[df_clean['cluster'] != -1]
    df_clean[column] = df_clean[column].rolling(window=window, center=True).mean()
    df_clean = df_clean.dropna(subset=[column])
    return df_clean


def calculate_period(signal, name, k_min_ratio=0.001, k_max=8640):
    k_min = max(1, int(len(signal) * k_min_ratio))
    lags = np.arange(k_min, k_max + 1)
    S = [np.sum((signal[k:] - signal[:-k]) ** 2) for k in lags]
    idx = np.argmin(S)
    T_auto = lags[idx]

    # Отрисовка полной кривой S(k)
    plt.figure(figsize=(10, 4))
    plt.plot(lags, S, label="S(k) — сумма квадратов ошибок", color="gray", alpha=0.5)

    # Отрисовка окна вокруг минимума
    delta = min(200, len(lags) // 2)
    start, end = max(0, idx - delta), min(len(lags), idx + delta)
    plt.plot(lags[start:end], np.array(S)[start:end], color="red", linewidth=2, label="Окно ±δ")
    plt.axvline(T_auto, color="blue", linestyle='--', label=f"Авто T = {T_auto}")
    plt.title(f"Поиск периода временного ряда — {name}")
    plt.xlabel("Сдвиг (k)")
    plt.ylabel("S(k)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Подтверждение или ручной ввод
    choice = input(f"Подтвердите период (Enter) или введите свой T вручную: ")
    if choice.strip().isdigit():
        T_manual = int(choice)
        print(f"Выбран вручную: T = {T_manual}")
        return T_manual
    print(f"Используется автоматически найденный T = {T_auto}")
    return T_auto


# Градиентный шаг, менеджеры остаются без изменений…
def gradient_step(x_prev, x_period, x_prev_period, x_curr, s, r, mu, alpha=1.0,
                  spike_thr=None, mu_spike=None, spike_repeats=0):
    if mu is None: mu = 1e-4
    if mu_spike is None: mu_spike = mu
    x_pred = s * x_prev + r * x_period - s * r * x_prev_period
    error = x_curr - x_pred
    mu_eff = mu * (1 + alpha * abs(error))
    grad_s = -error * (x_prev - r * x_prev_period)
    grad_r = -error * (x_period - s * x_prev_period)
    s_new, r_new = s - mu_eff * grad_s, r - mu_eff * grad_r
    # spike logic omitted…
    return s_new, r_new, error


class PseudoGradientManager:
    def __init__(self, T, mu, alpha, spike_thr, mu_spike, spike_repeats):
        self.T = T
        self.mu = mu
        self.alpha = alpha
        self.spike_thr = spike_thr
        self.mu_spike = mu_spike
        self.spike_repeats = spike_repeats
        self.s = 0.5
        self.r = 0.5

    def fit(self, signal):
        errors = []
        for t in range(self.T + 1, len(signal)):
            self.s, self.r, err = gradient_step(
                signal[t - 1], signal[t - self.T], signal[t - self.T - 1],
                signal[t], self.s, self.r,
                self.mu, self.alpha,
                self.spike_thr, self.mu_spike, self.spike_repeats
            )
            errors.append(err ** 2)
        return np.mean(errors)

    def predict(self, history):
        x_prev = history[-1]
        x_period = history[-self.T]
        x_prev_period = history[-self.T - 1]
        return self.s * x_prev + self.r * x_period - self.s * self.r * x_prev_period


class XGBManager:
    def __init__(self, T, **xgb_params):
        self.T = T
        self.model = XGBRegressor(**xgb_params)
        self.is_trained = False

    def fit(self, X, y):
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X):
        return self.model.predict(X)


def evaluate(true_vals, preds):
    mse = mean_squared_error(true_vals, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_vals, preds)
    mape = np.mean(np.abs((true_vals - preds) / true_vals)) * 100
    smape = np.mean(2 * np.abs(true_vals - preds) / (np.abs(true_vals) + np.abs(preds))) * 100
    r2 = r2_score(true_vals, preds)
    naive_err = np.mean(np.abs(np.diff(true_vals)))
    mase = mae / naive_err
    print(f"MSE={mse:.5f}, RMSE={rmse:.5f}, MAE={mae:.5f}")
    print(f"MAPE={mape:.2f}%, sMAPE={smape:.2f}%")
    print(f"R^2={r2:.4f}, MASE={mase:.3f}")


# ——————————————
#        MAIN
# ——————————————
if __name__ == '__main__':
    # 1) Загрузка и чистка
    df = load_data("VKR_dataset_test.csv")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    col = menu(numeric_cols)

    print("Предобработка данных…")
    df_clean = full_preprocessing(df, col)
    clean = df_clean[col].values
    raw = df[col].dropna().values

    # 2) Разделяем clean на train и test
    test_size = int(input("Доля тестовой выборки (в процентах, рек.20): ") or 20)
    n_test = max(1, int(len(clean) * test_size / 100))
    train, test = clean[:-n_test], raw[-n_test:]

    # 3) Поиск периода по train
    print("Поиск периода…")
    T = calculate_period(train, col, k_min_ratio=0.001, k_max=720)

    # 4) Настройка гиперпараметров (можно спросить у пользователя)
    mu, alpha = 1e-4, 1.0
    spike_thr = None
    mu_spike = None
    spike_reps = 0
    print(f"Параметры: mu={mu}, alpha={alpha}, spike_thr={spike_thr}, repeats={spike_reps}\n")

    # 5) Инициализируем менеджеры
    pg = PseudoGradientManager(T, mu, alpha, spike_thr, mu_spike, spike_reps)
    xgbm = XGBManager(T)

    # 6) Учим псевдоградиент на train
    print("Обучение псевдоградиента на train…")
    train_mse = pg.fit(train)
    train_rmse = np.sqrt(train_mse)
    print(f" s={pg.s:.5f}, r={pg.r:.5f}, RMSE_train={train_rmse:.5f}\n")

    # 7) Прогнозируем на длину test
    history = list(train)
    preds = []
    for i in range(len(test)):
        # если есть «скачок» — можно сюда вставить логику swticher, но для простоты — всегда pg
        x_next = pg.predict(history)
        preds.append(x_next)
        # теперь в историю добавляем не свой прогноз, а истинное test[i]
        history.append(test[i])

    # 8) Оцениваем качество (test vs preds)
    print("=== Диагностика на тесте ===")
    evaluate(test, np.array(preds))

    # 9) Рисуем четкий график
    plt.figure(figsize=(12, 4))
    # a) train
    plt.plot(np.arange(len(train)), train, label='Train', color='blue')
    # b) test (истинные)
    idx0 = len(train)
    plt.plot(np.arange(idx0, idx0 + len(test)), test, label='Test (real)', color='green')
    # c) прогноз
    plt.plot(np.arange(idx0, idx0 + len(preds)), preds, label='Forecast', color='orange')
    plt.axvline(idx0, color='gray', linestyle='--')
    plt.title(f'Прогноз {col} (Train/Test split)')
    plt.xlabel('Шаги')
    plt.ylabel(col)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
