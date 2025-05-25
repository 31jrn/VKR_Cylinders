import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from scipy.stats import zscore  # для вычисления Z-оценки
from sklearn.cluster import DBSCAN  # кластеризация и удаление выбросов


def load_data(file_path):
    df = pd.read_excel(file_path)
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
    df_clean.to_excel("cleaned_data.xlsx", index=False)
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


def calculate_period(signal, selected_column, k_min_ratio=0.001, k_max=500):
    k_min = int(len(signal) * k_min_ratio)
    S = []
    for k in range(k_min, k_max + 1):
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
            return min_k


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
    plt.show()


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
    plt.show()


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


def main():
    df = load_data("VKR_dataset_test.xlsx")
    columns = df.select_dtypes(include=["number"]).columns.tolist()
    selected_column = menu(columns)

    df_clean = data_preprocessing(df, selected_column)
    df.reset_index(drop=True, inplace=True)
    df_clean.reset_index(drop=True, inplace=True)
    period_signal = df_clean[selected_column].to_numpy()
    signal = df[selected_column].to_numpy()
    print(df.shape)
    print(df_clean.shape)

    # Вызов графика для сравнения обработанных и исходных данных
    orig_series = df[selected_column]
    clean_series = df_clean[selected_column]
    data_comparison(orig_series, clean_series, selected_column)

    T_period = calculate_period(period_signal, selected_column)
    build_spiral(signal, T_period, selected_column)
    model = build_regression(signal, T_period)
    original_data = signal[-(T_period + 1):]
    steps = int(input("Введите количество прогнозируемых значений: "))
    forecasted_data = forecast(model, original_data, T_period, steps)
    print(f"Прогноз на следующие 20 значений", forecasted_data)
    comparison_plot(original_data, forecasted_data, T_period)
    true_values = signal[-steps:]
    min_len = min(len(true_values), len(forecasted_data))
    evaluate_forecast(true_values[:min_len], forecasted_data[:min_len])


if __name__ == "__main__":
    main()
