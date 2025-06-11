import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.cluster import DBSCAN
from sklearn.metrics import mean_squared_error, mean_absolute_error
import sys
from sklearn.metrics import r2_score
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor


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


def hampel_fltr(signal, column, k, n_sigma):
    # Медиана в окне
    window_median = signal.rolling(window=2 * k + 1, center=True).median()
    abs_dev = (signal - window_median).abs()
    med_abs_dev = abs_dev.rolling(window=2 * k + 1, center=True).median()
    threshold = n_sigma * med_abs_dev  # Порог выбросов
    deviations = (signal[column] - window_median[column]).abs()
    outliers = deviations > threshold[column]
    cleaned_signal = signal.copy()
    cleaned_signal[outliers] = window_median[outliers]
    return cleaned_signal


def full_preprocessing(df, column, window=3, z_thresh=3.0):
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


def soft_preprocessing(signal, column, window=5):
    incoming_data = signal.dropna(subset=[column]).copy()
    soft_clean_signal = hampel_fltr(incoming_data, column, window, n_sigma=3.0)
    return soft_clean_signal


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
            x_prev = signal[t - 1]
            x_period = signal[t - self.T]
            x_prev_period = signal[t - self.T - 1]
            x_curr = signal[t]
            self.s, self.r, err = gradient_step(x_prev, x_period, x_prev_period, x_curr, self.s, self.r, self.mu,
                                                self.alpha, self.spike_thr, self.mu_spike, self.spike_repeats)
            errors.append(err ** 2)
        mean_sqr_err = sum(errors) / len(errors)
        return mean_sqr_err

    def predict(self, last_signal):
        x_prev = last_signal[-1]
        x_period = last_signal[-self.T]
        x_prev_period = last_signal[-self.T - 1]
        return self.s * x_prev + self.r * x_period - self.s * self.r * x_prev_period

    def update(self, x_real, last_signal):
        x_prev = last_signal[-2]
        x_period = last_signal[-1 - self.T]
        x_prev_period = last_signal[-2 - self.T]
        self.s, self.r, _ = gradient_step(x_prev, x_period, x_prev_period, x_real, self.s, self.r, self.mu, self.alpha,
                                          self.spike_thr, self.mu_spike, self.spike_repeats)


class XGBManager():
    def __init__(self, T, **xgb_params):
        self.T = T
        self.model = XGBRegressor()
        self.is_trained = False

    def fit(self, x, y):
        self.model.fit(x, y)
        self.is_trained = True

    def predict(self, x_frcst):
        if not self.is_trained:
            raise RuntimeError("Модель XGB еще не подходит")

        return self.model.predict(x_frcst)


class ModelSwitch:
    def __init__(self, window_size=20, thresh=2.0, quarantine=50):
        self.window = window_size
        self.thresh = thresh
        self.quarantine = quarantine
        self.errors = []  # список последних ошибок
        self.on_backup = False  # сейчас на XGB
        self.quarantine_left = 0  # сколько шагов ещё осталось работать

    def record_error(self, err):
        self.errors.append(abs(err))
        if len(self.errors) > self.window:
            self.errors.pop(0)

    def should_switch_to_backup(self):
        # Если ещё не на backup, считаем волатильность и сравниваем с thresh
        if self.on_backup:
            return False
        if len(self.errors) < self.window:
            return False
        vol = np.std(self.errors)
        if vol > self.thresh:
            self.on_backup = True
            self.quarantine_left = self.quarantine
            return True
        return False

    def should_return_to_main(self):
        # Если мы на backup, уменьшаем карантинный счётчик, и когда он исчерпан — обратная замена
        if not self.on_backup:
            return False
        self.quarantine_left -= 1
        if self.quarantine_left <= 0:
            self.on_backup = False
            return True
        return False


def gradient_step(x_prev, x_period, x_prev_period, x_curr, s, r, mu, alpha=1.0,
                  spike_thr=None, mu_spike=None, spike_repeats=0):
    x_pred = s * x_prev + r * x_period - s * r * x_prev_period
    error = x_curr - x_pred
    mu_eff = mu * (1 + alpha * abs(error))


def gradient_step(x_prev, x_period, x_prev_period, x_curr, s, r, mu, alpha=1.0,
                  spike_thr=None, mu_spike=None, spike_repeats=0):
    # --- ЗАЩИТА ОТ None ---
    if mu is None:
        mu = 1e-4
    if mu_spike is None:
        mu_spike = mu
    x_pred = s * x_prev + r * x_period - s * r * x_prev_period
    error = x_curr - x_pred
    mu_eff = mu * (1 + alpha * abs(error))
    grad_s = -error * (x_prev - r * x_prev_period)
    grad_r = -error * (x_period - s * x_prev_period)
    s_new, r_new = s - mu_eff * grad_s, r - mu_eff * grad_r
    if spike_thr is not None and abs(error) > spike_thr:
        for _ in range(spike_repeats):
            s_new, r_new, _ = gradient_step(x_prev, x_period, x_prev_period, x_curr, s_new, r_new, mu_spike or mu_eff,
                                            alpha, None, None, 0)
    return s_new, r_new, error


def features_targets(signal, history, T):
    X = [[history[i - 1], history[i - T]] for i in range(T, len(history))]
    y = [history[i] for i in range(T, len(history))]


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
    if len(true_vals) != len(preds):
        raise ValueError(f"Несоответствие длин: true={len(true_vals)}, preds={len(preds)}")
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
    # --------------------------
    # 1) Загрузка и предобработка
    df = load_data("VKR_dataset_test.csv")
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    col = menu(numeric_cols)

    print('Предобработка данных...')
    df_clean = full_preprocessing(df, col)
    clean = df_clean[col].values
    raw = df[col].dropna().values

    plot_comparison(df[col].dropna(), df_clean[col], col)

    # --------------------------
    # 2) Поиск периода
    print('Поиск периода...')
    T = calculate_period(clean, col, k_min_ratio=0.001, k_max=720)

    # --------------------------
    # 3) Настройка параметров
    mu, alpha = 1e-4, 1.0
    spike_thr = None
    mu_spike = None
    spike_reps = 0

    max_avail = len(raw) - len(clean)
    print(f"Параметры: mu={mu}, alpha={alpha}, spike_thr={spike_thr}, repeats={spike_reps}\n")

    steps = int(input('Шагов прогноза (рек.50): ') or 50)
    if steps > max_avail:
        print(f"ВНИМАНИЕ: запрошено {steps}, а доступно только {max_avail}. Будем прогнозировать {max_avail}.")
        steps = max_avail

    inp = input("Порог спайка (Enter — без спайка): ")
    spike_thr = float(inp) if inp else None
    inp = input("mu_spike (Enter — как mu_eff): ")
    mu_spike = float(inp) if inp else None
    inp = input("Число повторов при спайке (рек.0): ")
    spike_reps = int(inp) if inp.isdigit() else 0

    # --------------------------
    # 4) Инициализация менеджеров
    pg = PseudoGradientManager(T, mu=1e-5, alpha=1.0, spike_thr=spike_thr, mu_spike=mu_spike, spike_repeats=spike_reps)
    xgb = XGBManager(T)
    switcher = ModelSwitch(window_size=20, thresh=2 * 0.0, quarantine=50)
    # (thresh=2*train_rmse мы ещё не знаем — заполним позже)

    # --------------------------
    # 5) «Обучение» псевдоградиента и вычисление порога
    print('Обучение псевдоградиента...')
    train_mse = pg.fit(clean)
    train_rmse = np.sqrt(train_mse)
    switcher.thresh = 2 * train_rmse
    print(f"s={pg.s:.5f}, r={pg.r:.5f}, MSE_train={train_mse:.5f} (RMSE={train_rmse:.5f})\n")

    # --------------------------
    # 6) Онлайн-прогноз
    history = list(clean)
    preds_full = []
    real_idx = len(clean)

    for i in range(steps):
        # 1) прогноз из нужного менеджера
        if switcher.on_backup:
            # формируем единственную строку фич для XGB
            feat = np.array([[history[-1], history[-T]]])
            x_pred = xgb.predict(feat)[0]
        else:
            x_pred = pg.predict(history)

        preds_full.append(x_pred)

        # 2) действительное значение
        x_real = raw[real_idx + i]
        history.append(x_real)

        # 3) считаем ошибку
        err = x_real - x_pred
        switcher.record_error(err)

        # 4) если пора прыгнуть на XGB — обучаем его и «включаем»
        if not switcher.on_backup and switcher.should_switch_to_backup():
            # готовим X и y из всей накопленной истории
            X_train = []
            y_train = []
            for t in range(T, len(history)):
                X_train.append([history[t - 1], history[t - T]])
                y_train.append(history[t])
            xgb.fit(np.array(X_train), np.array(y_train))
            print(f"Перешли на XGB на шаге {i + 1} (волатильность {np.std(switcher.errors):.3f})")

        # 5) если назад хотим вернуться — switcher сам задаст on_backup=False
        switcher.should_return_to_main()

        # 6) если мы всё ещё на псевдоградиенте — делаем автоапдейт
        if not switcher.on_backup:
            pg.update(x_real, history)

    # --------------------------
    # 7) Диагностика
    true_vals = raw[len(clean): len(clean) + len(preds_full)]
    print("=== Диагностика прогноза ===")
    print("Requested steps:", steps)
    print("Switch to XGB happened:", switcher.on_backup is True or switcher.quarantine_left < 50)
    print("Total preds_full:", len(preds_full))
    evaluate(true_vals, np.array(preds_full))

    # --------------------------
    # 8) Итоговый график
    plt.figure(figsize=(12, 4))
    # а) исторические данные
    plt.plot(np.arange(len(raw)), raw, label='История', color='blue')
    # б) прогноз
    plt.plot(np.arange(len(clean), len(clean) + len(preds_full)), preds_full, label='Прогноз', color='orange')
    # в) реальные если хотим продолжение
    plt.plot(np.arange(len(clean), len(clean) + len(preds_full)), true_vals,
             label='Реальные продолжение', color='green', alpha=0.6, linestyle='--')
    plt.axvline(len(clean), color='gray', linestyle=':')
    plt.title(f'Прогноз {col}')
    plt.xlabel('Шаги')
    plt.ylabel(col)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
