import pandas as pd
import numpy as np
import matplotlib.pyplot
import mpl_toolkits.mplot3d
import matplotlib.pyplot as plt
from cProfile import label
from mpl_toolkits.mplot3d import Axes3D
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

if __name__ == "__main__":
    # Загрузка данных
    df = pd.read_csv("smoothed_data.csv")

    # Извлечение нужных признаков
    columns = ["TP2", "TP3", "H1", "DV_pressure", "Reservoirs", "Motor_current", "Oil_temperature"]
    X = df[columns].to_numpy()
    print(X.shape)


    def menu():
        print(f"Выберите характеристику для анализа: ")
        for id_x, col_name in enumerate(columns):
            print(f"{id_x}: {col_name}")
        print(f"{len(columns)}. Для выхода из программы")

        while True:
            choice = input("Введите номер анализируемой характеристики: ")
            try:
                selected_index = int(choice)
                if selected_index == len(columns):
                    print("Работа программы завершена")
                    break
                elif 0 <= selected_index < len(columns):
                    selected_column = columns[selected_index]
                    print(f"Вы выбрали {selected_column} для анализа")
                    return selected_index
                    break
                else:
                    print("Ошибка ввода. Номер вне допустимого диапазона. Попробуйте снова.")
            except ValueError:
                print("Ошибка. Введите целое число. Попробуйте снова.")


    # Реализация спиралевидного изображения
    number = menu()  # Передаваемое значение признака
    t_x = np.arange(X.shape[0])  # Массив значений
    signal = X[:, number]  # Цифра выбирает анализируемый признак
    k_min = int(X.shape[0] * 0.001)
    k_max = 500
    # Расчет суммы квадратов ошибок
    S = []
    for k in range(k_min, k_max + 1):
        shifted = signal[k:]
        original = signal[:-k]
        sqr_error_sum = np.sum((shifted - original) ** 2)
        S.append(sqr_error_sum)
    min_k = np.argmin(S) + k_min


    def get_validated_period(auto_T, min_allowed=30):
        print(f"Автоматически найденный период: T = {auto_T}")
        # График для проверки минимального периода
        k_values = np.arange(k_min, k_max + 1)
        plt.figure(figsize=(10, 4))
        plt.plot(k_values, S, label="Ошибка S(k)")
        plt.xlabel("Сдвиг (k)")
        plt.ylabel("Сумма квадратов ошибок")
        plt.title("Поиск периода временного ряда")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

        # Проверка подозрительно малого периода
        if auto_T < min_allowed:
            print(f"Период меньше минимально допустимого ({min_allowed}).")
        else:
            choice = input("Подтвердите период (Enter), или введите новое значение T вручную: ")
            if choice.strip() == "":
                return auto_T  # пользователь согласен
            else:
                try:
                    manual_T = int(choice)
                    print(f"Выбран вручную: T = {manual_T}")
                    return manual_T
                except ValueError:
                    print("Ошибка: введено не целое число. Используется авто-значение.")

        return auto_T


    T = get_validated_period(auto_T=min_k, min_allowed=30)

    # Построение цилиндрической поверхности(спиралей)
    """def cylinder_build(radius)
        print("Great")"""

    N = signal.shape[0]
    time_index = np.arange(N)
    radius = 1.0
    theta = 2 * np.pi * (t_x % T) / T
    z = t_x // T  # виток вдоль оси Z
    x_circle = radius * np.cos(theta)
    y_circle = radius * np.sin(theta)
    # Визуализация данных
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_circle, y_circle, z, color='blue', linewidth=0.5, label="Спираль")
    ax.scatter(x_circle, y_circle, z, c=signal, cmap='viridis', s=5, label="Значения")
    ax.set_title("Цилиндрическая модель временного ряда")
    ax.set_xlabel("X (cos)")
    ax.set_ylabel("Y (sin)")
    ax.set_zlabel("Виток (Z)")
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Регрессионная модель
    # Инициализация данных
    x_train = []
    y_train = []
    for t_count in range(T, len(signal)):
        x_prev = signal[t_count - 1]
        x_circ = signal[t_count - T]
        x_curr = signal[t_count]
        x_train.append([x_prev, x_circ])
        y_train.append(x_curr)

    x_train = np.array(x_train)
    y_train = np.array(y_train)
    # Метод наименьших квадратов
    model = LinearRegression()
    model.fit(x_train, y_train)
    print("Коэффициенты модели:", model.coef_)
    print("Свободный член ε:", model.intercept_)
    y_pred = model.predict(x_train)

    # Метрики качества
    mse = mean_squared_error(y_train, y_pred)
    rmse = mse ** 0.5
    coef_determ = r2_score(y_train, y_pred)
    print(f"Среднеквадратичная ошибка (RMSE) = {rmse}")
    print(f"Коэффициент детерминации R^2 = {coef_determ}")

