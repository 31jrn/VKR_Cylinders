import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.cluster import DBSCAN
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression


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


def data_preprocessing(df, column, window=3, z_thresh=3.0):
    # 1) Убираем NaN
    df_clean = df.dropna(subset=[column]).copy()
    # 2) Фильтрация по Z-оценке
    df_clean = df_clean[np.abs(zscore(df_clean[column])) < z_thresh]
    # 3) Удаление выбросов DBSCAN
    X = np.hstack([np.arange(len(df_clean)).reshape(-1, 1), df_clean[column].values.reshape(-1, 1)])
    labels = DBSCAN(eps=20, min_samples=5).fit_predict(X)
    df_clean['cluster'] = labels
    df_clean = df_clean[df_clean['cluster'] != -1]
    # 4) Скользящее среднее
    df_clean[column] = df_clean[column].rolling(window=window, center=True).mean()
    df_clean = df_clean.dropna(subset=[column])
    return df_clean


def plot_comparison(original, cleaned, name):
    plt.figure(figsize=(12, 4))
    plt.plot(original.index, original.values, label='Original data', alpha=0.4, color='gray')
    plt.plot(cleaned.index, cleaned.values, label='Cleaned data', color='blue')
    plt.title(f'Сравнение данных — {name}')
    plt.xlabel('Индекс')
    plt.ylabel(name)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


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


def gradient_step(x_prev, x_period, x_prev_period, x_curr, s, r, mu, alpha=1.0,
                  spike_thr=None, mu_spike=None, spike_repeats=0):
    x_pred = s * x_prev + r * x_period - s * r * x_prev_period
    error = x_curr - x_pred
    mu_eff = mu * (1 + alpha * abs(error))
    grad_s = -error * (x_prev - r * x_prev_period)
    grad_r = -error * (x_period - s * x_prev_period)
    s_new, r_new = s - mu_eff * grad_s, r - mu_eff * grad_r
    if spike_thr and abs(error) > spike_thr:
        for _ in range(spike_repeats):
            s_new, r_new, _ = gradient_step(x_prev, x_period, x_prev_period, x_curr, s_new, r_new, mu_spike or mu_eff,
                                            alpha, None, None, 0)
    return s_new, r_new, error


def predict_one(history, s, r, period):
    x_prev = history[-1]
    x_period = history[-period]
    x_prev_period = history[-period - 1]
    return s * x_prev + r * x_period - s * r * x_prev_period


def learn_pseudogradient(signal, T, mu=1e-4, alpha=1.0,
                         spike_thr=None, mu_spike=None, spike_repeats=0):
    s, r = 0.5, 0.5
    errors = []
    for t in range(T + 1, len(signal)):
        s, r, err = gradient_step(
            signal[t - 1], signal[t - T], signal[t - T - 1], signal[t],
            s, r, mu, alpha, spike_thr, mu_spike, spike_repeats
        )
        errors.append(err ** 2)
    return s, r, np.mean(errors)


def forecast_pseudogradient(signal, T, s, r, steps):
    history, preds = list(signal), []
    for _ in range(steps):
        x_prev, x_period, x_prev_period = history[-1], history[-T], history[-T - 1]
        x_next = s * x_prev + r * x_period - s * r * x_prev_period
        preds.append(x_next)
        history.append(x_next)
    return np.array(preds)


def evaluate(true_vals, preds):
    mse = mean_squared_error(true_vals, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_vals, preds)
    print(f"MSE={mse:.5f}, RMSE={rmse:.5f}, MAE={mae:.5f}")
    # MAPE
    mape = np.mean(np.abs((true_vals - preds) / true_vals)) * 100
    # sMAPE
    smape = np.mean(2 * np.abs(true_vals - preds) / (np.abs(true_vals) + np.abs(preds))) * 100
    print(f'MAPE: {mape:.2f}%    sMAPE: {smape:.2f}%')
    r2 = r2_score(true_vals, preds)
    print(f'R^2: {r2:.4f}')
    naive_err = np.mean(np.abs(np.diff(true_vals)))
    mase = np.mean(np.abs(true_vals - preds)) / naive_err
    print(f'MASE: {mase:.3f}')


def build_regression(signal, T):
    X, y = [], []
    for t in range(T, len(signal)):
        X.append([signal[t - 1], signal[t - T]])
        y.append(signal[t])
    X = np.array(X)
    y = np.array(y)
    model = LinearRegression().fit(X, y)
    print(f"Regression coef: {model.coef_}, intercept: {model.intercept_:.5f}")
    return model


def forecast_regression(model, history, T, steps):
    preds = []
    buf = history.copy()
    for _ in range(steps):
        x_prev, x_T = buf[-1], buf[-T]
        y_next = model.predict([[x_prev, x_T]])[0]
        preds.append(y_next)
        buf.append(y_next)
    return np.array(preds)


if __name__ == '__main__':
    df = load_data(file_path="VKR_dataset_test.csv")
    cols = df.select_dtypes(include=[np.number]).columns.tolist()
    col = menu(cols)

    # 1) Предобработка
    print('Предобработка данных...')
    df_clean = data_preprocessing(df, col)
    clean = df_clean[col].values
    raw = df[col].dropna().values
    # Сравнение до и после очистки
    plot_comparison(df[col].dropna(), df_clean[col], col)

    # 2) Поиск периода на очищенных данных
    print('Поиск периода...')
    T = calculate_period(clean, col, k_min_ratio=0.001, k_max=720)
    print(f'Используем период T={T}\n')

    # 3) Рекомендуемые параметры
    mu, alpha, spike_thr, mu_spike, spike_reps = 1e-4, 1.0, None, None, 0
    max_available = len(raw) - len(clean)
    print(f"Параметры: mu={mu}, alpha={alpha}, spike_thr={spike_thr}, repeats={spike_reps}\n")
    # ввод желаемого горизонта
    steps = int(input('Шагов прогноза (рек.50): ') or 50)
    if steps > max_available:
        print(
            f"ВНИМАНИЕ: Вы запрашиваете {steps} шагов, но в raw-данных доступно только {max_available} для сравнения.")
        steps = max_available
        print(f"Шагов прогнозирования ограничено до: {steps}")

    s_thr = input("Порог спайка (Enter — отключить): ")
    spike_thr = float(s_thr) if s_thr else None
    m_sp = input("mu_spike (Enter — как mu_eff): ")
    mu_spike = float(m_sp) if m_sp else None
    rep = input("Число повторов при спайке (рек.0): ")
    spike_repeats = int(rep) if rep.isdigit() else 0

    # 4) Обучение и прогноз
    print('Обучение псевдоградиента...')
    s, r, train_mse = learn_pseudogradient(clean, T, mu, alpha, spike_thr, mu_spike, spike_reps)
    train_rmse = np.sqrt(train_mse)
    print(f's={s:.5f}, r={r:.5f}, MSE_train={train_mse:.5f}\n')
    print('Прогнозирование (online-режим)...')
    preds = []
    history = list(raw)

    # задаём пороги
    E_thr = 5 * train_rmse  # например, вдвое больше RMSE на обучении
    max_consec = 5  # сколько подряд больших ошибок терпим

    consec_bad = 0  # счётчик подряд «плохих» шагов
    preds_pg = []  # прогноз псевдоградиентом

    for i in range(steps):
        # 1) прогноз
        x_pred = predict_one(history, s, r, T)
        preds_pg.append(x_pred)

        # 2) реальное
        x_real = raw[len(clean) + i]
        history.append(x_real)

        # 3) вычисление ошибки
        err = abs(x_real - x_pred)

        # 4) обновление счётчика
        if err > E_thr:
            consec_bad += 1
        else:
            consec_bad = 0

        # 5) проверка условия переключения
        if consec_bad >= max_consec:
            switch_idx = i + 1
            print(f"Switching to regression at step {switch_idx} (error={err:.3f})")
            break

        # 6) онлайн-апдейт параметров
        s, r, _ = gradient_step(history[-2], history[-1 - T], history[-2 - T], x_real, s, r, mu, alpha, spike_thr,
                                mu_spike, spike_repeats
                                )
    else:
        switch_idx = steps  # не переключились

    reg_model = build_regression(clean, T)
    preds_reg = forecast_regression(reg_model, history, T, steps - switch_idx)

    if switch_idx < steps:
        preds_full = np.concatenate([preds_pg, preds_reg])
    else:
        preds_full = np.array(preds_pg)
    print("=== Диагностика прогноза ===")
    print("Requested steps:", steps)
    print("Switch index:", switch_idx)
    print("Length of online preds (preds_pg):", len(preds_pg))
    print("Length of regression preds (preds_reg):", len(preds_reg))
    print("Total preds_full:", len(preds_full))

    # 1) Готовим ось X для исторических сырых данных, но только до конца clean:
    x_hist = np.arange(len(clean))  # индексы от 0 до (len(clean)-1)
    y_hist = raw[:len(clean)]  # первые len(clean) точек из raw

    # 2) Готовим ось X для прогноза:
    x_fore = np.arange(len(clean), len(clean) + len(preds_full))
    y_fore = preds_full  # здесь уже вся длина прогноза

    # 3) Рисуем
    plt.figure(figsize=(10, 4))

    # 3.1) Сырые данные на историческом участке
    plt.plot(x_hist, y_hist,
             label='Original data',
             color='blue', alpha=0.4)

    # 3.2) Собственно прогноз
    plt.plot(x_fore, y_fore,
             'r--', linewidth=1,
             label='Forecast')
    plt.scatter(x_fore, y_fore,
                c='red', s=20,
                label='Forecast points')

    # 3.3) Если есть точка переключения — вертикальная линия
    if switch_idx < steps:
        plt.axvline(x=len(clean) + switch_idx,
                    color='gray', linestyle=':',
                    label='Switch point')

    plt.title(f'Forecast for {col}')
    plt.xlabel('Step index')
    plt.ylabel(col)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
