import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from mpl_toolkits.mplot3d import Axes3D


def load_data(file_path):
    df = pd.read_csv(file_path)
    return df


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


def main():
    df = load_data("smoothed_data.csv")
    columns = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Motor_current", "Oil_temperature"]
    selected_column = menu(columns)
    signal = df[selected_column].to_numpy()

    T_period = calculate_period(signal, selected_column)
    build_spiral(signal, T_period, selected_column)
    build_regression(signal, T_period)


if __name__ == "__main__":
    main()
