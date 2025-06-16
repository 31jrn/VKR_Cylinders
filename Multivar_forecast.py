import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.signal import medfilt, butter, filtfilt
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from typing import List
from pandas import DataFrame
from sklearn.model_selection import TimeSeriesSplit
from scipy.fft import fft, fftfreq

warnings.filterwarnings("ignore", category=FutureWarning)


# Загрузка
def load_data(fp):
    return pd.read_csv(fp, sep=';')


# Автокалибровка
def calibrate(signal):
    v = signal
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

def low_pass(signal, cutoff=0.4, fs=1.0, order=2):
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

    # На «smooth» калибруем (единожды) параметры классификатора
    if not calibrated:
        params = calibrate(raw)
        calibrated = True

    smooth = low_pass(raw, cutoff=0.4, fs=1.0, order=2)
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


def multipreprocessing(df: pd.DataFrame, cols: List[str]) -> DataFrame:
    out = pd.DataFrame(index=df.index)
    for c in cols:
        raw, smooth, proc, sig_type = preprocess(df, c)
        # если гибрид — берём background, иначе берём proc
        clean = proc if sig_type != 'hybrid' else proc['background']
        out[c] = clean
    return out


def scale_data(df: DataFrame):
    scaler = MinMaxScaler()
    arr = scaler.fit_transform(df.values)
    return arr, scaler


class CylindricalFeaturizer:
    """
    Класс для извлечения признаков из цилиндрического представления временных рядов.
    """

    def __init__(self):
        self.periods_ = {}  # Словарь для хранения найденных периодов

    @staticmethod
    def find_period_autocorr(signal: np.ndarray, k_max_ratio=0.5) -> int:
        """
        Находит период, минимизируя сумму квадратов разностей.
        Это неинтерактивная версия вашей функции.
        """
        n = len(signal)
        k_min = 2
        k_max = int(n * k_max_ratio)

        # Чтобы избежать слишком больших k_max на малых данных
        if k_max < k_min:
            return 10  # Возвращаем значение по умолчанию

        lags = np.arange(k_min, k_max)
        s = [np.mean((signal[k:] - signal[:-k]) ** 2) for k in lags]

        # Находим первый локальный минимум, чтобы избежать кратных периодов
        min_idx = -1
        for i in range(1, len(s) - 1):
            if s[i - 1] > s[i] and s[i] < s[i + 1]:
                min_idx = i
                break

        if min_idx == -1:  # если минимумов не найдено
            min_idx = np.argmin(s)

        return lags[min_idx]

    @staticmethod
    def find_period_fft(signal: np.ndarray, fs: float = 1.0) -> int:
        """
        Находит доминирующий период с помощью быстрого преобразования Фурье (FFT).
        """
        n = len(signal)
        if n < 2: return 10

        yf = fft(signal)
        xf = fftfreq(n, 1 / fs)

        # Ищем пик в положительной части частотного спектра
        # Игнорируем нулевую частоту (постоянную составляющую)
        idx = np.argmax(np.abs(yf[1:n // 2])) + 1
        dominant_freq = xf[idx]

        if dominant_freq == 0:
            return n  # Если нет доминирующей частоты, берем всю длину

        return int(1 / dominant_freq)

    def transform_and_extract(self, window: np.ndarray, period: int) -> np.ndarray:
        """
        Преобразует окно данных в цилиндрические координаты и извлекает признаки.
        """
        if period <= 1: period = len(window)

        t = np.arange(len(window))

        # 1. Цилиндрическое преобразование
        # Угол 'phi' зависит от времени, высота 'z' - от значения сигнала
        phi = (t % period) * (2 * np.pi / period)
        z = window

        # 2. Извлечение признаков
        # Статистики по высоте (значению)
        z_mean = np.mean(z)
        z_std = np.std(z)

        # Статистики по углу (чтобы избежать проблем с переходом через 2*PI)
        # Они характеризуют, в какой "части" цикла сейчас находится система
        cos_phi_mean = np.mean(np.cos(phi))
        sin_phi_mean = np.mean(np.sin(phi))

        # Локальный тренд внутри окна
        # Коэффициент наклона линейной регрессии
        trend = np.polyfit(t, z, 1)[0]

        # Объединяем все признаки в один вектор
        return np.array([z_mean, z_std, cos_phi_mean, sin_phi_mean, trend])

    def fit_transform(self, df: pd.DataFrame, lag: int, period_method: str = 'autocorr') -> pd.DataFrame:
        """
        Применяет извлечение цилиндрических признаков ко всему датафрейму.
        """
        all_cyl_features = []
        feature_names = []

        for col_name in df.columns:
            signal = df[col_name].values

            # Находим период для всего сигнала
            if period_method == 'fft':
                period = self.find_period_fft(signal)
            else:  # 'autocorr' по умолчанию
                period = self.find_period_autocorr(signal)
            self.periods_[col_name] = period
            print(f"Найден период T={period} для признака '{col_name}' методом '{period_method}'")

            # Извлекаем признаки для каждого окна
            col_features = []
            for i in range(len(signal) - lag + 1):
                window = signal[i: i + lag]
                features = self.transform_and_extract(window, period)
                col_features.append(features)

            # Собираем DataFrame признаков для текущего столбца
            base_feature_names = ['z_mean', 'z_std', 'cos_phi', 'sin_phi', 'trend']
            current_feature_names = [f"{col_name}_cyl_{name}" for name in base_feature_names]

            # Добавляем в общий список
            all_cyl_features.append(pd.DataFrame(col_features, columns=current_feature_names))

        # Объединяем все DataFrame с цилиндрическими признаками в один
        return pd.concat(all_cyl_features, axis=1)


def create_multivariate_samples(data: np.ndarray, n_in: int, n_out: int, target_idx: int):
    X, y = [], []
    L, n_feats = data.shape
    for i in range(L - n_in - n_out + 1):
        win = data[i: i + n_in, :]  # shape (n_in, n_feats)
        X.append(win.flatten())  # (n_in * n_feats,)
        future = data[i + n_in: i + n_in + n_out, target_idx]
        y.append(future[0] if n_out == 1 else future)
    return np.array(X), np.array(y)


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


def cross_validate_series(X, y, n_splits=5, **xgb_kwargs):
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    tscv = TimeSeriesSplit(n_splits=n_splits)

    metrics = {
        'rmse': [], 'mae': [], 'r2': [], 'mape': [], 'smape': [],
        'mase': [], 'logcosh': [], 'wape': [], 'rmsle': []
    }
    last_model = None
    last_true = None
    last_pred = None

    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X[train_idx], X[test_idx]
        y_tr, y_val = y[train_idx], y[test_idx]

        model = XGBRegressor(**xgb_kwargs)
        model.fit(X_tr, y_tr,
                  eval_set=[(X_tr, y_tr), (X_val, y_val)],
                  verbose=False)

        y_pred = model.predict(X_val)

        if np.any(np.isnan(y_pred)) or np.any(np.isnan(y_val)):
            continue

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        with np.errstate(divide='ignore', invalid='ignore'):
            nonzero_mask = y_val != 0
            if np.any(nonzero_mask):
                mape = np.mean(np.abs((y_val[nonzero_mask] - y_pred[nonzero_mask]) / y_val[nonzero_mask])) * 100
            else:
                mape = np.nan
            smape = np.mean(2 * np.abs(y_val - y_pred) / (np.abs(y_val) + np.abs(y_pred))) * 100
        naive_err = np.mean(np.abs(np.diff(y_val)))
        mase = np.mean(np.abs(y_val - y_pred)) / naive_err
        log_cosh = np.log(np.cosh(y_pred - y_val))
        log_cosh_loss = np.mean(log_cosh)
        wape = np.sum(np.abs(y_val - y_pred)) / np.sum(np.abs(y_val))
        rmsle = np.sqrt(np.mean((np.log1p(y_val) - np.log1p(y_pred)) ** 2))

        metrics['rmse'].append(rmse)
        metrics['mae'].append(mae)
        metrics['r2'].append(r2)
        metrics['mape'].append(mape)
        metrics['smape'].append(smape)
        metrics['mase'].append(mase)
        metrics['logcosh'].append(log_cosh_loss)
        metrics['wape'].append(wape)
        metrics['rmsle'].append(rmsle)

        last_model = model
        last_true = y_val
        last_pred = y_pred

    avg = {
        k: {
            'mean': float(np.mean(v)),
            'std': float(np.std(v))
        } for k, v in metrics.items()
    }
    return avg, last_model, last_true, last_pred


def print_metrics_table(column: str, avg_metrics: dict, n_splits: int):
    print(f"\n=== {column} (CV over {n_splits} folds) ===")
    keys = list(avg_metrics.keys())
    for i in range(0, len(keys), 2):
        row = []
        for j in range(2):
            if i + j >= len(keys):
                continue
            k = keys[i + j]
            m = avg_metrics[k]['mean']
            s = avg_metrics[k]['std']
            name = k.upper()
            if k in ['mape', 'smape']:
                row.append(f"{name}: {m:.2f}% ± {s:.2f}%")
            elif k == 'logcosh':
                row.append(f"{name}: {m:.5f} ± {s:.5f}")
            else:
                row.append(f"{name}: {m:.4f} ± {s:.4f}")
        print(" | ".join(row))


def train_and_evaluate_multivariate(scaled_data, scaler, columns, lag=20, test_ratio=0.8, n_splits=5):
    n_feats = scaled_data.shape[1]
    results = {}

    # Генерация имён фичей (можно вынести в __main__)
    feat_names = []
    for l in range(1, lag + 1):
        for name in columns:
            feat_names.append(f"{name}_t-{l}")

    for tgt_idx, column in enumerate(columns):
        X, y = create_multivariate_samples(scaled_data, n_in=lag, n_out=1, target_idx=tgt_idx)
        # вместо простого train/test — CV:
        avg_metrics, model, y_true, y_pred = cross_validate_series(X, y, n_splits=5, n_estimators=500,
                                                                   learning_rate=0.01, early_stopping_rounds=15,
                                                                   verbosity=0)
        print_metrics_table(column, avg_metrics, n_splits)

        results[column] = {'model': model, 'y_true': y_true, 'y_pred': y_pred, 'metrics': avg_metrics}

    return results


# Основной
if __name__ == '__main__':
    df = pd.read_csv('VKR_dataset_test.csv', sep=';')

    # 1. Список признаков
    using_columns = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature', 'Motor_current']

    # 2. «Грубая» предобработка
    clean_df = multipreprocessing(df, using_columns)
    LAG = 20  # Задаем глубину истории

    # 3. ✨ ИЗВЛЕЧЕНИЕ ЦИЛИНДРИЧЕСКИХ ПРИЗНАКОВ (НОВЫЙ ШАГ) ✨
    featurizer = CylindricalFeaturizer()
    # Вы можете выбрать метод 'fft' или 'autocorr'
    cyl_features_df = featurizer.fit_transform(clean_df, lag=LAG, period_method='autocorr')

    # 4. ОБЪЕДИНЕНИЕ ДАННЫХ
    # Убедимся, что индексы совпадают для корректного объединения
    # clean_df должен быть достаточно длинным, чтобы из него можно было извлечь признаки
    end_idx = len(cyl_features_df)
    combined_df = pd.concat([clean_df.iloc[:end_idx].reset_index(drop=True), cyl_features_df], axis=1)

    # 5. Масштабирование ОБЪЕДИНЕННОГО датафрейма
    scaled_data, scaler = scale_data(combined_df)
    # обучение + оценка
    results = train_and_evaluate_multivariate(scaled_data, scaler, combined_df.columns, lag=LAG, test_ratio=0.8,n_splits=5)

    for column, result in results.items():
        if 'y_true' in result and 'y_pred' in result:
            plt.figure(figsize=(8, 3))
            plt.plot(result['y_true'], label='True', linewidth=1.5)
            plt.plot(result['y_pred'], label='Predicted', linestyle='--')
            plt.title(f"Prediction vs True: {column}")
            plt.xlabel("Time")
            plt.ylabel(column)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
