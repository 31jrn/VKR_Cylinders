import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from scipy.signal import butter, filtfilt
import math
import sys


# Загрузка
def load_data(fp):
    return pd.read_csv(fp, sep=';')


# Автокалибровка
def calibrate(signal):
    v = signal  # уже NumPy‑массив
    return {
        'unique_thresh': 0.05,
        'epsilon': np.std(np.diff(v)) * 0.01,
        'plateau_thresh': 0.5,
        'jump_factor': 5,
        'jump_ratio_thresh': 0.1,
        'n_states': min(len(np.unique(v)), 4),
        'long_win': 60,
        'var_thresh': np.std(v) * 0.1,
        'bg_win': 3,
        'event_thr': np.std(np.diff(v)) * 3
    }


# Классификация

def low_pass(signal, cutoff=0.2, fs=1.0, order=3):
    """
    Низкочастотный фильтр Баттерворта:
      - cutoff — нормированная граничная частота (0 < cutoff < 1),
                 например 0.2 — пропустит колебания ниже 0.2·Nyquist.
      - fs     — частота дискретизации (1.0, если данные равномерны по времени).
      - order  — порядок фильтра (2–4 обычно достаточно).
    """
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, signal)


def classify(signal, p):
    frac = len(np.unique(signal)) / len(signal)
    d = np.abs(np.diff(signal))
    if frac < p['unique_thresh'] or (d < p['epsilon']).mean() > p['plateau_thresh']:
        return 'discrete'
    if (d > p['jump_factor'] * np.median(d)).mean() > p['jump_ratio_thresh']:
        return 'hybrid'
    return 'continuous'


# Обработка
def process_discrete(signal, p):
    k = p['n_states']
    km = KMeans(n_clusters=k).fit(signal.reshape(-1, 1))
    centers = np.sort(km.cluster_centers_.flatten())
    labels = km.predict(signal.reshape(-1, 1))
    # вместо метки — значение кластера
    return centers[labels]


def process_continuous(signal, p):
    out = np.zeros_like(signal)
    for i in range(len(signal)):
        lv = np.std(signal[max(0, i - p['long_win']):i + 1])
        w = 5 if lv > p['var_thresh'] else p['long_win']
        a, b = max(0, i - w // 2), min(len(signal), i + w // 2)
        out[i] = np.mean(signal[a:b])
    return out


def process_hybrid(signal, p):
    bg = medfilt(signal, kernel_size=p['bg_win'])
    d = np.diff(signal, prepend=signal[0])
    ev = [(int(i), float(d[i])) for i in np.where(np.abs(d) > p['event_thr'])[0]]
    return {'background': bg, 'events': ev}


# Предобработка
calibrated = False
params = {}


def preprocess(df, column):
    global calibrated, params
    raw = df[column].dropna().values

    # Применяем low‑pass — получаем «гладкий» сигнал
    smooth = low_pass(raw, cutoff=0.2, fs=1.0, order=3)

    # На «smooth» калибруем (единожды) параметры классификатора
    if not calibrated:
        params = calibrate(raw)
        calibrated = True

    smooth = low_pass(raw, cutoff=0.2, fs=1.0, order=3)
    t = classify(smooth, params)

    # Классифицируем по smooth
    t = classify(smooth, params)

    # Обрабатываем — опять же по smooth
    if t == 'discrete':
        proc = process_discrete(smooth, params)
    elif t == 'continuous':
        proc = process_continuous(smooth, params)
    else:
        proc = process_hybrid(smooth, params)

    # Возвращаем и исходный raw (для графиков/прогноза), и smooth, и proc
    return raw, smooth, proc, t


# Визуализация
def viz(raw, smooth, proc, t, column):
    plt.figure(figsize=(12, 4))
    plt.plot(raw, alpha=0.3, label='raw')
    plt.plot(smooth, alpha=0.6, label='smooth')
    if signal_type == 'discrete':
        plt.step(range(len(processed_signal)), processed_signal, where='mid', label='states')
    elif signal_type == 'continuous':
        plt.plot(processed_signal, label='smoothed')
    else:
        plt.plot(processed_signal['background'], label='bg')
        for i, _ in processed_signal['events']: plt.axvline(i, color='r', ls='--')
    plt.title(f"{column} — {signal_type}")
    plt.legend()
    plt.show()


# Определение периода
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


# Меню выбора столбца
def menu(columns):
    for i, c in enumerate(columns): print(f"{i}: {c}")
    print(f"{len(columns)}: Выход из программы")
    while True:
        ch = input("Выберите признак: ")
        if ch.isdigit() and int(ch) <= len(columns):
            idx = int(ch)
            if idx == len(columns): sys.exit()
            return columns[idx]


# Делим данные
def split(clean, raw, frac=0.1):
    n = len(clean)
    nt = int(n * frac)
    return clean[:-nt], raw[-nt:]


# ПсевдоГрадиент
class PGM:
    def __init__(self, T, mu=1e-5, alpha=1):
        self.T, self.mu, self.alpha = T, mu, alpha
        self.s, self.r = 0.5, 0.5

    def fit(self, signal):
        errs = []
        for t in range(self.T + 1, len(signal)):
            x0, x1, x2, x = signal[t - 1], signal[t - self.T], signal[t - self.T - 1], signal[t]
            p = self.s * x0 + self.r * x1 - self.s * self.r * x2
            e = x - p
            mu = self.mu * (1 + self.alpha * abs(e))
            gs = -e * (x0 - self.r * x2)
            gr = -e * (x1 - self.s * x2)
            self.s -= mu * gs
            self.r -= mu * gr
            errs.append(e * e)
        return np.mean(errs)

    def predict(self, hist):
        return self.s * hist[-1] + self.r * hist[-self.T] - self.s * self.r * hist[-self.T - 1]

    def update(self, x, hist):
        pred = self.predict(hist)
        e = x - pred
        mu = self.mu * (1 + self.alpha * abs(e))
        x0, x1, x2 = hist[-2], hist[-1 - self.T], hist[-2 - self.T]
        gs = -e * (x0 - self.r * x2)
        gr = -e * (x1 - self.s * x2)
        self.s -= mu * gs
        self.r -= mu * gr


# Переключатель
class Switch:
    def __init__(self, w=20, th=0, qt=50):
        self.w, self.th, self.qt = w, th, qt
        self.errs = []
        self.on = False
        self.q = 0

    def rec(self, e):
        self.errs.append(abs(e))
        if len(self.errs) > self.w: self.errs.pop(0)

    def to_backup(self):
        if self.on or len(self.errs) < self.w: return False
        if np.std(self.errs) > self.th:
            self.on = True
            self.q = self.qt
            return True
        return False

    def to_main(self):
        if not self.on: return False
        self.q -= 1
        if self.q <= 0:
            self.on = False
            return True
        return False


# XGB-менеджер
class XGBM:
    def __init__(self, T):
        self.T = T
        self.m = XGBRegressor()
        self.tr = False

    def fit(self, X, y): self.m.fit(X, y);self.tr = True

    def predict(self, F):
        if not self.tr: raise RuntimeError
        return self.m.predict(F)


# Оценивание прогноза


def evaluate(true_vals, preds):
    true_vals = np.asarray(true_vals)
    preds = np.asarray(preds)
    if true_vals.shape != preds.shape:
        raise ValueError(f"Несоответствие длин: true={true_vals.shape}, preds={preds.shape}")
    mse = mean_squared_error(true_vals, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_vals, preds)
    print(f"MSE={mse:.5f}, RMSE={rmse:.5f}, MAE={mae:.5f}")
    # MAPE
    with np.errstate(divide='ignore', invalid='ignore'):
        mape = np.mean(np.abs((true_vals - preds) / true_vals)) * 100
    print(f"MAPE: {mape:.2f}%", end='    ')
    # sMAPE
    denom = np.abs(true_vals) + np.abs(preds)
    with np.errstate(divide='ignore', invalid='ignore'):
        smape = np.mean(2 * np.abs(true_vals - preds) / denom) * 100
    print(f"sMAPE: {smape:.2f}%")
    r2 = r2_score(true_vals, preds)
    print(f"R^2: {r2:.4f}")
    naive_err = np.mean(np.abs(np.diff(true_vals)))
    mase = np.mean(np.abs(true_vals - preds)) / naive_err
    print(f"MASE: {mase:.3f}")
    # Log‐Cosh Loss (векторизованный)
    log_cosh = np.log(np.cosh(preds - true_vals))
    log_cosh_loss = np.mean(log_cosh)
    print(f"Log‐Cosh Loss: {log_cosh_loss:.5f}")
    # WAPE
    wape = np.sum(np.abs(true_vals - preds)) / np.sum(np.abs(true_vals))
    print(f"WAPE: {wape:.5f}")


# Основной
if __name__ == '__main__':
    df = load_data('VKR_dataset_test.csv')
    column = menu(df.select_dtypes(include=[np.number]).columns)
    raw = df[column].dropna().values
    raw, smooth, processed_signal, signal_type = preprocess(df, column)
    #  например, для графика покажем и smooth
    viz(raw, smooth, processed_signal, signal_type, column)
    clean = processed_signal if signal_type != 'hybrid' else processed_signal['background']
    train, test = split(clean, raw)
    T = calculate_period(processed_signal, column)
    pg = PGM(T)
    x0 = pg.fit(train)
    rmse = np.sqrt(x0)
    sw = Switch(th=2 * rmse)
    xb = XGBM(T)
    hist = list(train)
    preds = []
    for i, x in enumerate(test):
        if sw.on:
            F = np.array([[hist[-1], hist[-T]]])
            y = xb.predict(F)[0]
        else:
            y = pg.predict(hist)
        preds.append(y)
        err = x - y
        sw.rec(err)
        if not sw.on and sw.to_backup():
            Xtr = [[hist[t - 1], hist[t - T]] for t in range(T, len(hist))]
            ytr = [hist[t] for t in range(T, len(hist))]
            xb.fit(np.array(Xtr), np.array(ytr))
            print(f"switch at {i}")
        if not sw.on: pg.update(x, hist)
        hist.append(x)

    evaluate(test, preds)
    plt.figure(figsize=(12, 4))
    plt.plot(raw, alpha=0.5, label='hist')
    plt.plot(range(len(train), len(train) + len(preds)), preds, label='fcst')
    plt.axvline(len(train), ls='--')
    plt.legend()
    plt.show()
