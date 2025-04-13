import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# === 1. Чтение данных ===
file_path = "smoothed_data.csv"  # Данные после очистки и сглаживания
data = pd.read_csv(file_path)

# Определяем временной ряд, например, TP2
series = data["TP2"].values


# === 2. Функция для построения цилиндрической модели ===
def build_cylindrical_model(series, period):
    """
    Построение цилиндрической модели временного ряда.
    :param series: временной ряд.
    :param period: период временного ряда.
    :return: двумерный массив цилиндрической развёртки.
    """
    n = len(series)
    num_cycles = n // period
    model = series[:num_cycles * period].reshape(num_cycles, period)
    return model


# === 3. Функция обновления параметров с использованием псевдоградиентного метода ===
def update_model_params(cylindrical_model, alpha=0.01, num_iterations=100):
    """
    Обновление параметров модели с использованием псевдоградиентного метода.
    :param cylindrical_model: двумерный массив цилиндрической развёртки.
    :param alpha: шаг обучения.
    :param num_iterations: количество итераций.
    :return: обновлённые параметры модели.
    """
    num_cycles, period = cylindrical_model.shape
    # Инициализация параметров модели
    params = np.random.rand(period)

    for iteration in range(num_iterations):
        gradients = np.zeros_like(params)
        for k in range(num_cycles - 1):
            error = cylindrical_model[k + 1] - np.dot(cylindrical_model[k], params)
            gradients += -2 * cylindrical_model[k] * error
        params -= alpha * gradients / (num_cycles - 1)
    return params


# === 4. Прогнозирование ===
def predict_next(cylindrical_model, params):
    """
    Прогноз следующего значения временного ряда.
    :param cylindrical_model: двумерный массив цилиндрической развёртки.
    :param params: параметры модели.
    :return: прогнозируемое значение.
    """
    last_cycle = cylindrical_model[-1]
    return np.dot(last_cycle, params)


# === 5. Основной алгоритм ===
# Задаём период временного ряда (например, 24 для часовых данных)
period = 24
cylindrical_model = build_cylindrical_model(series, period)

# Обновляем параметры модели
params = update_model_params(cylindrical_model)
print("Обновлённые параметры модели:", params)

# Прогнозируем следующее значение
next_value = predict_next(cylindrical_model, params)
print("Прогнозируемое следующее значение:", next_value)

# === 6. Визуализация ===
plt.plot(series, label="Оригинальный временной ряд")
plt.axhline(y=next_value, color='r', linestyle='--', label="Прогноз")
plt.legend()
plt.title("Цилиндрическая модель и прогноз")
plt.show()
