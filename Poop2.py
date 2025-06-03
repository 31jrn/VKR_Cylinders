import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import zscore  # для вычисления Z-оценки
from sklearn.cluster import DBSCAN  # кластеризация и удаление выбросов
from statsmodels.tsa.stattools import acf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import time



def load_data(file_path):
    df = pd.read_csv(file_path, sep=';')
    return df


def data_preprocessing(df, column, window=3, z_thresh=3.0):
    # Очистка пустых строк
    df_clean = df.dropna(subset=[column])
    # Очистка по Z-оценке
    z_scores = zscore(df_clean[column])
    df_clean = df_clean[np.abs(z_scores) < z_thresh]
    # DBSCAN
    timestamps = np.arange(len(df_clean)).reshape(-1, 1)
    values = df_clean[column].values.reshape(-1, 1)
    X = np.hstack([timestamps, values])
    db_clas = DBSCAN(eps=20, min_samples=5)
    db_clas.fit(X)
    labels = db_clas.labels_
    df_clean['cluster'] = labels
    df_clean = df_clean[labels != -1]
    # Скользящее среднее
    df_clean[column] = df_clean[column].rolling(window=window, center=True).mean()
    df_clean = df_clean.dropna(subset=[column])
    return df_clean


def menu(columns):
    print("\nВыберите характеристику для анализа:")
    for idx, col_name in enumerate(columns):
        print(f"{idx}: {col_name}")
    print(f"{len(columns)}: Выход из программы")

    while True:
        choice = input("\nВведите номер характеристики: ")
        try:
            selected_index = int(choice)
            if selected_index == len(columns):
                print("Выход из программы.")
                exit()  # завершить программу
            elif 0 <= selected_index < len(columns):
                selected_column = columns[selected_index]
                print(f"Вы выбрали {selected_column} для анализа.\n")
                return selected_column
            else:
                print("Номер вне диапазона. Попробуйте снова.")
        except ValueError:
            print("Ошибка: введите целое число.")


def calculate_period(signal, selected_column, k_min_ratio=0.001, k_max=8640, interactive=True):
    k_min = int(len(signal) * k_min_ratio)
    lags = np.arange(k_min, k_max + 1)
    S = [np.sum((signal[k:] - signal[:-k]) ** 2) for k in lags]

    """for k in range(k_min, k_max + 1):
        shifted = signal[k:]
        original = signal[:-k]
        sqr_error_sum = np.sum((shifted - original) ** 2)
        S.append(sqr_error_sum)
    min_k = np.argmin(S) + k_min

    
    # Визуализация для пользователя
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(k_min, k_max + 1), S, label="Ошибка S(k)")
    plt.xlabel("Сдвиг (k)")
    plt.ylabel("Сумма квадратов ошибок")
    plt.title(f"Поиск периода временного ряда {selected_column}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
       # Предложить подтвердить/изменить
    if min_k < 30:
        print(f"Найденный период {min_k} подозрительно мал.")
    else:
        print(f"Автоматически найденный период: {min_k}")
    
    choice = input("Введите период вручную или нажмите Enter для подтверждения: ")
    if choice.strip() == "":
        return min_k
    else:
        try:
            manual_T = int(choice)
            print(f"Выбран вручную: {manual_T}")
            return manual_T
        except ValueError:
            print("Ошибка ввода. Используется автоматический период.")
            return min_k"""
    # Автоматический минимум
    idx_auto = np.argmin(S)
    k_auto = lags[idx_auto]

    # Отрисовка графика
    plt.figure(figsize=(10, 4))
    plt.plot(lags, S, color="gray", alpha=0.3, label="S(k) полный")
    delta = min(200, len(lags))
    start = max(0, idx_auto - delta)
    end = min(len(lags) - 1, idx_auto + delta)
    plt.plot(lags[start:end + 1], np.array(S)[start:end + 1], color="red", linewidth=2,
             label=f"Окно ±{delta} вокруг авто T={k_auto}")
    plt.axvline(k_auto, color="red", linestyle='--')
    plt.xlabel("Сдвиг k")
    plt.ylabel("S(k) Сумма квадратов ")
    plt.title(f"Поиск периода для {selected_column}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Вмешательство (ввод нужного значения)
    if interactive:
        choice = input(f"Автоматический T={k_auto}. Введите свой период T или нажмите Enter: ")
        if choice.strip().isdigit():
            k_manual = int(choice)
            print(f"Используется пользовательское значение периода: T = {k_manual}")
            return k_manual

    print(f"Используется автоматическое значение периода: {k_auto}")
    return k_auto


"""
def build_spiral(signal, T, selected_column, radius=1.0):
    N = signal.shape[0]
    t_x = np.arange(N)
    theta = 2 * np.pi * (t_x % T) / T
    z = t_x // T
    x_circle = radius * np.cos(theta)
    y_circle = radius * np.sin(theta)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_circle, y_circle, z, color='blue', linewidth=0.5, label="Спираль")
    scatter = ax.scatter(x_circle, y_circle, z, c=signal, cmap='viridis', s=5, label="Значения")
    ax.set_title(f"Цилиндрическая модель временного ряда {selected_column}")
    ax.set_xlabel("X (cos)")
    ax.set_ylabel("Y (sin)")
    ax.set_zlabel("Виток (Z)")
    ax.legend()
    plt.colorbar(scatter, ax=ax, label="Значения сигнала")
    plt.tight_layout()
    # plt.show()"""


def data_comparison(original_data: pd.Series, cleaned_data: pd.Series, column):
    plt.figure(figsize=(14, 6))
    plt.plot(original_data.index, original_data.values, label="Исходные данные", alpha=0.5, color="gray")
    plt.plot(cleaned_data.index, cleaned_data.values, label="Очищенные данные", color="blue")
    plt.title(f"Сравнение исходных и очищенных данных {column}")
    plt.xlabel("Индекс (время)")
    plt.ylabel("Значение сигнала")
    plt.legend(loc="upper left")
    plt.grid(True)
    plt.tight_layout()
    # plt.show()


def build_regression(signal, T):
    x_train = []
    y_train = []
    for t in range(T, len(signal)):
        x_prev = signal[t - 1]
        x_circ = signal[t - T]
        x_curr = signal[t]
        x_train.append([x_prev, x_circ])
        y_train.append(x_curr)

    x_train = np.array(x_train)
    y_train = np.array(y_train)

    model = LinearRegression()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_train)

    print("Коэффициенты модели:", model.coef_)
    print("Свободный член ε:", model.intercept_)
    mse = mean_squared_error(y_train, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_train, y_pred)
    print(f"Среднеквадратичная ошибка (RMSE) = {rmse:.5f}")
    print(f"Коэффициент детерминации R^2 = {r2:.5f}")
    return model


def forecast(model, signal, T, steps):
    predicted_data = []
    buffer = signal.copy().tolist()

    for step in range(steps):
        x_prev = buffer[-1]
        x_periodic = buffer[-T]
        x_forecast = np.array([[x_prev, x_periodic]])
        y_next = model.predict(x_forecast)[0]
        buffer.append(y_next)
        predicted_data.append(y_next)
    return np.array(predicted_data)


def comparison_plot(signal, predicted_data, T):
    # График сравнения реального тренда и прогноза
    plt.figure(figsize=(12, 6))
    plt.plot(range(len(signal)), signal, label="Исходные данные")
    forecast_range = range(len(signal), len(signal) + len(predicted_data))
    plt.plot(forecast_range, predicted_data, 'r--o', label="Прогнозируемые значения", markersize=3)
    plt.xlabel("Время (индексы)")
    plt.ylabel("Значение сигнала")
    plt.title("Прогноз значений на основе цилиндрической модели")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def seasonality_estimate(signal, max_lag, alpha=0.5, plot=False):
    data = np.array(signal)
    n = len(data)
    max_lag = max_lag or n // 3

    acf_values, confidence_interval = acf(data, nlags=max_lag, alpha=alpha)
    lags = np.arange(len(acf_values))

    # ищем первый «значимый» пик после лага 0
    # условимся: «значимый» = локальный максимум и > доверительного интервала
    # (либо просто > соседних)
    best_lag = None
    best_value = -np.inf
    for i in range(1, len(acf_values) - 1):
        if acf_values[i] > acf_values[i - 1] and acf_values[i] > acf_values[i + 1]:
            lower, upper = confidence_interval[i]
            if acf_values[i] > upper:
                best_value = acf_values[i]
                best_lag = i
    if best_lag is None:
        best_lag = int(np.argmax(acf_values[1:]) + 1)
    return best_lag, acf_values, lags


def seasonal_model(signal, T):
    model = SARIMAX(signal, order=(1, 0, 1), seasonal_order=(1, 1, 1, T), enforce_stationarity=False,
                    enforce_invertibility=False)
    fit_results = model.fit(disp=False, method='powell', maxiter=10)
    return fit_results

    # stepwise=True — быстрый перебор, n_jobs=-1 — все ядра
"""sarima = pm.auto_arima(
        signal,
        start_p=1, d=0, start_q=1,
        seasonal=True, m=T,
        start_P=1, D=1, start_Q=1,
        max_p=1, max_q=1,
        max_P=1, max_Q=1,
        stepwise=True,
        n_jobs=-1,
        suppress_warnings=True,
        error_action='ignore'
    )
    return sarima"""


def evaluate_forecast(true_values, predicted_values):
    mse = mean_squared_error(true_values, predicted_values)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_values, predicted_values)

    # Предотвращение деления на ноль в MAPE
    non_zero_indices = true_values != 0
    mape = np.mean(np.abs(
        true_values[non_zero_indices] - predicted_values[non_zero_indices] / true_values[non_zero_indices])) * 100
    print("\n Метрики качества прогноза:")
    print(f"MSE (дисперсия ошибок): {mse:.5f}")
    print(f"RMSE (среднеквадратичная ошибка): {rmse:.5f}")
    print(f"MAE (средний модуль отклонения): {mae:.5f}")
    print(f"MAPE (процентная ошибка): {mape:.2f}%")


# Определение псевдоградиента и функции ошибки
def gradient_step(x_prev, x_period, x_prev_period, x_curr, s, r, mu):
    e_pred_error = x_curr - (s * x_prev + r * x_period - s * r * x_prev_period)
    gradient_s = - e_pred_error * (x_prev - r * x_prev_period)
    gradient_r = - e_pred_error * (x_period - s * x_prev_period)

    s_new = s - mu * gradient_s
    r_new = r - mu * gradient_r
    return s_new, r_new, e_pred_error


def learn_pseudogradient(signal, period, learning_rate):
    s_param, r_param = 0.5, 0.5
    errors = []
    T = period
    for t in range(T + 1, len(signal)):
        x_prev = signal[t - 1]
        x_period = signal[t - T]
        x_prev_period = signal[t - T - 1]
        x_current = signal[t]
        # Один шаг псевдоградиента
        s_param, r_param, error = gradient_step(x_prev, x_period, x_prev_period, x_current, s_param, r_param,
                                                learning_rate)
        errors.append(error ** 2)
    return s_param, r_param, np.mean(errors)


def forecast_pseudeogradient(signal, period, s, r, steps):
    history = signal.copy().tolist()
    predictions = []
    T = period
    for _ in range(steps):
        t = len(history)
        x_previous = history[-1]
        x_period = history[-T]
        x_prev_period = history[-T - 1]
        # Модель Хабиби
        x_next = s * x_previous + r * x_period - s * r * x_prev_period
        predictions.append(x_next)
        history.append(x_next)
    return np.array(predictions)


def main():
    df = load_data("VKR_dataset_test_10t.csv")
    columns = df.select_dtypes(include=["number"]).columns.tolist()
    selected_column = menu(columns)

    print(">>> START data_preprocessing", time.time())
    df_clean = data_preprocessing(df, selected_column)
    df.reset_index(drop=True, inplace=True)
    df_clean.reset_index(drop=True, inplace=True)
    period_signal = df_clean[selected_column].to_numpy()
    signal = df[selected_column].to_numpy()
    signal = signal[:1500]
    print(df.shape)
    print(df_clean.shape)
    print("<<< END data_preprocessing", time.time())

    # Вызов графика для сравнения обработанных и исходных данных
    orig_series = df[selected_column]
    clean_series = df_clean[selected_column]
    print(">>> START data_comparison", time.time())
    data_comparison(orig_series, clean_series, selected_column)
    print("<<< END data_comparison", time.time())
    print(">>> START calculate_period", time.time())
    T_period = calculate_period(signal, selected_column, k_min_ratio=0.001, k_max=1440, interactive=True)
    print(f"Оцененный сезонный период T = {T_period}")
    print("<<< END calculate_period", time.time())
    # SARIMA
    print(">>> START seasonal_model", time.time())
    sarima_res = seasonal_model(signal, T_period)
    print("<<< END seasonal_model", time.time())
    # построим «аппроксимацию» (in-sample):
    print(">>> START sarima_res.fittedvalues", time.time())
    in_sample = sarima_res.fittedvalues
    plt.figure()
    plt.plot(signal, label="Исходный ряд")
    plt.plot(in_sample, label="SARIMA in-sample", alpha=0.7)
    plt.legend()
    plt.title("SARIMA: in-sample fit")
    plt.show()
    print("<<< END sarima_res.fittedvalues", time.time())

    # Обучение параметров псевдоградиента
    print(">>> START learn_pseudogradient", time.time())
    s_param, r_param, mse_train = learn_pseudogradient(signal, T_period, 1e-4)
    print("Выученные параметры:", s_param, r_param)
    print("MSE на обучении:", mse_train)
    print("<<< END learn_pseudogradient", time.time())

    # Прогноз на будущие шаги(steps)
    steps = int(input("Введите количество прогнозируемых значений: "))
    print(">>> START forecast_pseudeogradient", time.time())
    predictions = forecast_pseudeogradient(signal, T_period, s_param, r_param, steps)
    sarima_forecast = sarima_res.get_forecast(steps=steps)
    sarima_pred = sarima_forecast.predicted_mean  # это pd.Series длины steps
    sarima_conf_int = sarima_forecast.conf_int()  # для доверительных интервалов
    print("<<< END forecast_pseudeogradient", time.time())

    # График сравнения прогноза
    print(">>> START comparison_plot", time.time())
    comparison_plot(signal, predictions, T_period)
    true_values = signal[-steps:]
    min_len = min(len(true_values), steps)
    print("<<< END comparison_plot", time.time())

    print(">>> START sarima_forecast.predicted_mean", time.time())
    sarima_values = sarima_pred
    print("<<< END sarima_forecast.predicted_mean", time.time())

    print(">>> START evaluate_forecast", time.time())
    print("\n--- Pseudo-gradient forecast ---")
    evaluate_forecast(true_values[:min_len], predictions[:min_len])
    print("\n--- SARIMA forecast ---")
    evaluate_forecast(true_values[:min_len], sarima_pred[:min_len])
    print("<<< END evaluate_forecast", time.time())
    plt.figure(figsize=(12, 5))
    t = np.arange(len(signal))
    plt.plot(t, signal, label="История")
    plt.plot(t[-min_len:], predictions[-min_len:], 'r--', label="Pseudo-grad")
    plt.plot(t[-min_len:], sarima_pred[-min_len:], 'g-.', label="SARIMA")
    plt.legend()
    plt.title("Сравнение прогнозов")
    plt.show()


if __name__ == "__main__":
    main()
