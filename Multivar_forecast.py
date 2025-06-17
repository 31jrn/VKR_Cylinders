import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
from scipy.signal import medfilt, butter, filtfilt
# from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
from sklearn.preprocessing import MinMaxScaler
from typing import List, Dict, Tuple
from pandas import DataFrame
from sklearn.model_selection import TimeSeriesSplit
from scipy.fft import fft, fftfreq
from statsmodels.graphics.tsaplots import plot_acf
import seaborn as sns
import os
import multiprocessing

os.environ['LOKY_MAX_CPU_COUNT'] = str(multiprocessing.cpu_count())
warnings.filterwarnings("ignore", category=UserWarning, module='joblib')
warnings.filterwarnings("ignore", category=FutureWarning)


# --- Вспомогательные функции и классы ---

def load_data(fp: str) -> pd.DataFrame:
    """Загружает данные из CSV файла."""
    return pd.read_csv(fp, sep=';')


def calibrate(signal: np.ndarray) -> dict:
    """Автокалибровка параметров для классификации сигнала."""
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


def low_pass(signal: np.ndarray, cutoff: float = 0.4, fs: float = 1.0, order: int = 2) -> np.ndarray:
    """Низкочастотный фильтр Баттерворта."""
    b, a = butter(order, cutoff / (0.5 * fs), btype='low')
    return filtfilt(b, a, signal)


"""def classify(signal: np.ndarray, p: dict) -> str:
    #Классифицирует сигнал как 'discrete', 'hybrid' или 'continuous'
    frac = len(np.unique(signal)) / len(signal)
    d = np.abs(np.diff(signal))
    if frac < p['unique_thresh'] or (d < p['epsilon']).mean() > p['plateau_thresh']:
        return 'discrete'
    if (d > p['jump_factor'] * np.median(d)).mean() > p['jump_ratio_thresh']:
        return 'hybrid'
    return 'continuous'


def process_discrete(signal: np.ndarray, p: dict) -> np.ndarray:
    # Обработка дискретных сигналов с помощью KMeans
    k = p['n_states']
    km = KMeans(n_clusters=k, n_init=10).fit(signal.reshape(-1, 1)) # n_init добавлено для совместимости с новыми версиями sklearn
    centers = np.sort(km.cluster_centers_.flatten())
    labels = km.predict(signal.reshape(-1, 1))
    return centers[labels]


def process_continuous(signal: np.ndarray, p: dict) -> np.ndarray:
    #Обработка непрерывных сигналов путем скользящего среднего
    out = np.zeros_like(signal)
    for i in range(len(signal)):
        lv = np.std(signal[max(0, i - p['long_win']):i + 1])
        w = 5 if lv > p['var_thresh'] else p['long_win']
        a, b = max(0, i - w // 2), min(len(signal), i + w // 2)
        out[i] = np.mean(signal[a:b])
    return out


def process_hybrid(signal: np.ndarray, p: dict) -> dict:
    #Обработка гибридных сигналов (фон + события)
    bg = medfilt(signal, kernel_size=p['bg_win'])
    d = np.diff(signal, prepend=signal[0])
    ev = [(int(i), float(d[i])) for i in np.where(np.abs(d) > p['event_thr'])[0]]
    return {'background': bg, 'events': ev}"""

# Глобальные переменные для калибровки
_calibrated = False
_params = {}

"""def preprocess(df: pd.DataFrame, column: str):
    # Выполняет предобработку одного столбца DataFrame:
    # фильтрация, классификация и специализированная обработка.
    global _calibrated, _params
    raw = df[column].dropna().values

    if not _calibrated:
        _params = calibrate(raw)
        _calibrated = True

    smooth = low_pass(raw, cutoff=0.4, fs=1.0, order=2)
    sig_type = classify(smooth, _params)

    if sig_type == 'discrete':
        proc = process_discrete(smooth, _params)
    elif sig_type == 'continuous':
        proc = process_continuous(smooth, _params)
    else:  # 'hybrid'
        proc = process_hybrid(smooth, _params)

    return raw, smooth, proc, sig_type


def multipreprocessing(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    # Выполняет предобработку нескольких столбцов DataFrame.
    cleans = []
    for c in cols:
        raw, smooth, proc, sig_type = preprocess(df, c)
        clean = proc if sig_type != 'hybrid' else proc['background']
        cleans.append(clean)
    min_len = min(len(arr) for arr in cleans)
    out = pd.DataFrame({cols[i]: cleans[i][:min_len] for i in range(len(cols))})
    return out"""


def preprocess_simple(df: pd.DataFrame, column: str, use_median_filter: bool = True) -> np.ndarray:
    """
    Простая предобработка одного столбца DataFrame:
    просто медианная фильтрация или низкочастотная фильтрация.
    Возвращает очищенный массив numpy.
    """
    raw = df[column].dropna().values

    if use_median_filter:
        # Медианный фильтр хорошо удаляет выбросы, сохраняя края
        # Выбери подходящий размер окна. 3 - это минимум. Можно попробовать 5 или 7.
        clean_signal = medfilt(raw, kernel_size=3)
    else:
        # Низкочастотный фильтр Баттерворта (как в исходном low_pass)
        clean_signal = low_pass(raw, cutoff=0.4, fs=1.0, order=2)

    return clean_signal


def multipreprocessing_simple(df: pd.DataFrame, cols: List[str], use_median_filter: bool = True) -> pd.DataFrame:
    """Выполняет простую предобработку нескольких столбцов DataFrame."""
    cleans = []
    for c in cols:
        clean = preprocess_simple(df, c, use_median_filter=use_median_filter)
        cleans.append(clean)
    min_len = min(len(arr) for arr in cleans)
    out = pd.DataFrame({cols[i]: cleans[i][:min_len] for i in range(len(cols))})
    return out


def scale_data(df: DataFrame):
    """Масштабирует данные к диапазону [0, 1] и возвращает DataFrame."""
    scaler = MinMaxScaler()
    arr_scaled = scaler.fit_transform(df.values)
    scaled_df = pd.DataFrame(arr_scaled, columns=df.columns, index=df.index)
    return scaled_df, scaler


class CylindricalFeaturizer:
    """
    Класс для работы с квазипериодическими сигналами,
    включая поиск периода по методу минимизации суммы квадратов разностей
    и FFT (если нужен).
    """

    def __init__(self):
        self.periods_ = {}

    @staticmethod
    def find_period_autocorr(signal: np.ndarray, max_period_lag: int = 200, min_autocorr_threshold: float = 0.05,
                             return_errors: bool = False) -> Tuple[
        int, np.ndarray]:
        """
        Определяет квазипериод сигнала на основе максимума автокорреляционной функции.
        Может также возвращать значения автокорреляции для визуализации.
        """
        if len(signal) < 2:
            if return_errors:
                return 1, np.array([0.0])
            return 1

        # Нормализуем сигнал для ACF
        norm_signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-9)
        autocorr = np.correlate(norm_signal, norm_signal, mode='full')
        autocorr = autocorr[len(autocorr) // 2:]  # Берем только положительные лаги
        autocorr = autocorr / autocorr[0]  # Нормализуем до 1 (acf на лаге 0 всегда 1)

        # Диапазон лагов для поиска (исключаем лаг 0)
        search_range_autocorr = autocorr[1:min(len(autocorr), max_period_lag + 1)]

        if len(search_range_autocorr) == 0:
            if return_errors:
                return 1, np.array([0.0])
            return 1

        # Находим все локальные максимумы, которые выше порога
        significant_peaks_lags = []
        for i in range(1, len(search_range_autocorr) - 1):  # Начинаем с лага 1 (индекс 0 в search_range_autocorr)
            if (search_range_autocorr[i] > search_range_autocorr[i - 1] and
                    search_range_autocorr[i] > search_range_autocorr[i + 1] and
                    search_range_autocorr[i] > min_autocorr_threshold):
                significant_peaks_lags.append(i + 1)  # +1 потому что индексы массива начинаются с 0, а лаги с 1

        period = 1  # Дефолтное значение

        if significant_peaks_lags:
            period = significant_peaks_lags[0]
        else:
            period = np.argmax(search_range_autocorr) + 1  # +1 для корректного лага

        if return_errors:
            return period, search_range_autocorr
        return period

    @staticmethod
    def find_period_fft(signal: np.ndarray, fs: float = 1.0) -> int:
        """
        Находит доминирующий период с помощью быстрого преобразования Фурье (FFT).
        """
        n = len(signal)
        if n < 2: return 10

        yf = fft(signal)
        xf = fftfreq(n, 1 / fs)

        idx = np.argmax(np.abs(yf[1:n // 2])) + 1
        dominant_freq = xf[idx]

        if dominant_freq == 0:
            return 10

        return int(1 / dominant_freq)


def create_multivariate_samples(data: np.ndarray, n_in: int, n_out: int, target_idx: int):
    """
    Создает выборки для обучения модели прогнозирования:
    X - лаговые признаки, y - целевое значение.
    """
    X, y = [], []
    L, n_feats = data.shape
    for i in range(L - n_in - n_out + 1):
        win = data[i: i + n_in, :]
        X.append(win.flatten())
        future = data[i + n_in: i + n_in + n_out, target_idx]
        y.append(future[0] if n_out == 1 else future)
    return np.array(X), np.array(y)


def evaluate(true_vals: np.ndarray, preds: np.ndarray):
    """Выводит стандартные метрики оценки прогноза."""
    true_vals = np.asarray(true_vals)
    preds = np.asarray(preds)
    if true_vals.shape != preds.shape:
        raise ValueError(f"Несоответствие длин: true={true_vals.shape}, preds={preds.shape}")

    mse = mean_squared_error(true_vals, preds)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(true_vals, preds)
    print(f"MSE={mse:.5f}, RMSE={rmse:.5f}, MAE={mae:.5f}")

    with np.errstate(divide='ignore', invalid='ignore'):
        nonzero_mask = true_vals != 0
        mape = np.nan if not np.any(nonzero_mask) else np.mean(
            np.abs((true_vals[nonzero_mask] - preds[nonzero_mask]) / true_vals[nonzero_mask])) * 100
    print(f"MAPE: {mape:.2f}%", end='    ')

    denom = np.abs(true_vals) + np.abs(preds)
    with np.errstate(divide='ignore', invalid='ignore'):
        smape = np.mean(2 * np.abs(true_vals - preds) / denom) * 100
    print(f"sMAPE: {smape:.2f}%")

    r2 = r2_score(true_vals, preds)
    print(f"R^2: {r2:.4f}")

    naive_err = np.mean(np.abs(np.diff(true_vals)))
    mase = mae / naive_err if naive_err != 0 else np.nan
    print(f"MASE: {mase:.3f}")

    log_cosh = np.log(np.cosh(preds - true_vals))
    log_cosh_loss = np.mean(log_cosh)
    print(f"Log‐Cosh Loss: {log_cosh_loss:.5f}")

    wape = np.sum(np.abs(true_vals - preds)) / np.sum(np.abs(true_vals)) if np.sum(np.abs(true_vals)) != 0 else np.nan
    print(f"WAPE: {wape:.5f}")

    rmsle = np.sqrt(np.mean((np.log1p(true_vals) - np.log1p(preds)) ** 2)) if np.all(true_vals >= 0) and np.all(
        preds >= 0) else np.nan
    print(f"RMSLE: {rmsle:.4f}")  # Добавил сюда вывод RMSLE


def cross_validate_series(X: np.ndarray, y: np.ndarray, n_splits: int = 5, **xgb_kwargs):
    """
    Выполняет временную кросс-валидацию для модели XGBoost.
    Возвращает средние метрики, последнюю обученную модель, истинные и предсказанные значения.
    """
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
        model.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_val, y_val)], verbose=False)
        y_pred = model.predict(X_val)

        if np.any(np.isnan(y_pred)) or np.any(np.isnan(y_val)):
            continue

        try:
            rmse = mean_squared_error(y_val, y_pred, squared=False)
        except TypeError:
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))

        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)

        with np.errstate(divide='ignore', invalid='ignore'):
            nonzero_mask = y_val != 0
            mape = np.mean(np.abs((y_val[nonzero_mask] - y_pred[nonzero_mask]) / y_val[nonzero_mask])) * 100 if np.any(
                nonzero_mask) else np.nan
            smape = np.mean(2 * np.abs(y_val - y_pred) / (np.abs(y_val) + np.abs(y_pred))) * 100
        naive_err = np.mean(np.abs(np.diff(y_val)))
        mase = mae / naive_err if naive_err != 0 else np.nan
        log_cosh_loss = np.mean(np.log(np.cosh(y_pred - y_val)))
        wape = np.sum(np.abs(y_val - y_pred)) / np.sum(np.abs(y_val)) if np.sum(np.abs(y_val)) != 0 else np.nan
        rmsle = np.sqrt(np.mean((np.log1p(y_val) - np.log1p(y_pred)) ** 2)) if np.all(y_val >= 0) and np.all(
            y_pred >= 0) else np.nan

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
            'mean': float(np.mean(v)) if v else np.nan,
            'std': float(np.std(v)) if v else np.nan
        } for k, v in metrics.items()
    }
    return avg, last_model, last_true, last_pred


def print_metrics_table(column: str, avg_metrics: dict, n_splits: int):
    """Выводит отформатированную таблицу метрик."""
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
            elif k in ['logcosh', 'wape', 'rmsle']:  # Добавлен RMSLE
                row.append(f"{name}: {m:.5f} ± {s:.5f}")
            else:
                row.append(f"{name}: {m:.4f} ± {s:.4f}")
        print(" | ".join(row))


def train_and_evaluate_multivariate(scaled_data: pd.DataFrame, target_columns: List[str],
                                    all_feature_columns: List[str], lag: int = 20, n_splits: int = 5):
    """
    Обучает и оценивает модель XGBoost для многомерного прогнозирования
    с использованием кросс-валидации.
    """
    results = {}

    xgb_params = {
        'n_estimators': 600,
        'learning_rate': 0.01,
        'early_stopping_rounds': 10,
        'eval_metric': 'rmse',
        'verbosity': 0
    }

    full_input_feat_names = []
    for l in range(1, lag + 1):
        for name in all_feature_columns:
            full_input_feat_names.append(f"{name}_t-{l}")

    for target_col in target_columns:
        tgt_idx = scaled_data.columns.get_loc(target_col)

        X, y = create_multivariate_samples(scaled_data.values, n_in=lag, n_out=1, target_idx=tgt_idx)

        avg_metrics, model, y_true, y_pred = cross_validate_series(X, y, n_splits=n_splits, **xgb_params)

        print_metrics_table(target_col, avg_metrics, n_splits)

        feature_importances_df = pd.DataFrame({
            'Feature': full_input_feat_names,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        results[target_col] = {
            'model': model,
            'y_true': y_true,
            'y_pred': y_pred,
            'metrics': avg_metrics,
            'feature_importances': feature_importances_df
        }
    return results


def create_global_cylindrical_features(df: DataFrame, period_map: dict) -> DataFrame:
    """
    Создает цилиндрические признаки для каждого столбца в DataFrame на основе глобальных периодов.
    """
    cyl_df = df.copy()
    for col_name, T0 in period_map.items():
        if col_name in df.columns and T0 is not None and T0 > 0:
            time_points = np.arange(len(df))
            phi = (2 * np.pi / T0) * time_points

            cyl_df[f'{col_name}_cyl_cos_phi'] = np.cos(phi)
            cyl_df[f'{col_name}_cyl_sin_phi'] = np.sin(phi)

            cyl_df[f'{col_name}_cyl_z_std'] = df[col_name].rolling(window=int(T0), min_periods=1).std()
            cyl_df[f'{col_name}_cyl_z_mean'] = df[col_name].rolling(window=int(T0), min_periods=1).mean()

            cyl_df[f'{col_name}_cyl_z_std'] = cyl_df[f'{col_name}_cyl_z_std'].fillna(method='bfill').fillna(
                method='ffill')
            cyl_df[f'{col_name}_cyl_z_mean'] = cyl_df[f'{col_name}_cyl_z_mean'].fillna(method='bfill').fillna(
                method='ffill')
        elif col_name in df.columns:
            cyl_df[f'{col_name}_cyl_cos_phi'] = np.nan
            cyl_df[f'{col_name}_cyl_sin_phi'] = np.nan
            cyl_df[f'{col_name}_cyl_z_std'] = np.nan
            cyl_df[f'{col_name}_cyl_z_mean'] = np.nan
            print(
                f"  Предупреждение: Период для '{col_name}' не найден или некорректен. Цилиндрические признаки будут NaN.")
    return cyl_df


def plot_cylindrical_representation(signal: np.ndarray, T0: float, title: str = "Цилиндрическое представление",
                                    R: float = 1.0):
    """
    Строит 3D-график временного ряда на цилиндре.
    """
    if T0 is None or T0 <= 0:
        print(f"Невозможно построить цилиндрическое представление для '{title}': некорректный период T0={T0}.")
        return

    t = np.arange(len(signal))

    X = R * np.cos((2 * np.pi / T0) * t)
    Y = R * np.sin((2 * np.pi / T0) * t)
    Z = signal

    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(X, Y, Z, label=f'Сигнал {title}', alpha=0.8)
    ax.set_xlabel('X (координата на цилиндре)')
    ax.set_ylabel('Y (координата на цилиндре)')
    ax.set_zlabel('Значение сигнала (Z)')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()


# --- Основной блок выполнения ---
if __name__ == '__main__':
    df = load_data("VKR_dataset_test.csv")

    using_columns_original = ['TP2', 'TP3', 'H1', 'DV_pressure', 'Reservoirs', 'Oil_temperature', 'Motor_current']
    LAG = 20

    print("Этап 1: Предобработка исходных данных...")
    # Используем упрощенную предобработку
    clean_df = multipreprocessing_simple(df, using_columns_original, use_median_filter=True)  # Попробуй True или False
    print("  Предобработка завершена.")

    print("\nЭтап 2: Оценка периодов квазипериодических процессов...")
    periods = {}
    for col in using_columns_original:
        # Получаем значения автокорреляции для построения графика
        # ИСПРАВЛЕНИЕ ЗДЕСЬ: используем CylindricalFeaturizer.find_period_autocorr
        period, autocorr_values = CylindricalFeaturizer.find_period_autocorr(clean_df[col].values, return_errors=True)
        periods[col] = period
        print(f"  Период для '{col}': {period} шагов")

        # Визуализация ACF
        plt.figure(figsize=(10, 4))
        plt.plot(np.arange(1, len(autocorr_values) + 1), autocorr_values)
        plt.title(f'Автокорреляционная функция для {col}')
        plt.xlabel('Лаг')
        plt.ylabel('Автокорреляция')
        plt.axvline(x=period, color='r', linestyle='--', label=f'Найденный период: {period}')
        plt.legend()
        plt.grid(True)
        plt.show()

    all_experiment_results = {}

    # --- Эксперимент 1: Прогноз только на исходных лаговых признаках ---
    print("\n--- ЭКСПЕРИМЕНТ 1: Прогноз только на исходных лаговых признаках ---")

    scaled_original_data, scaler_orig = scale_data(clean_df[using_columns_original])

    results_exp1 = train_and_evaluate_multivariate(
        scaled_original_data, using_columns_original, scaled_original_data.columns.tolist(),
        lag=LAG, n_splits=5
    )
    all_experiment_results['Без цилиндрических признаков'] = results_exp1

    # --- Эксперимент 2: Прогноз с использованием цилиндрических признаков ---
    print("\n--- ЭКСПЕРИМЕНТ 2: Прогноз с использованием цилиндрических признаков ---")

    df_with_cyl_features = create_global_cylindrical_features(clean_df, periods)

    # Важно: убедиться, что df_combined_for_exp2 содержит только numeric data.
    # Если какие-то колонки полностью NaN, они будут удалены.
    df_combined_for_exp2 = pd.concat([clean_df, df_with_cyl_features], axis=1)
    df_combined_for_exp2 = df_combined_for_exp2.dropna(axis=1, how='all')

    scaled_cyl_data, scaler_cyl = scale_data(df_combined_for_exp2)

    results_exp2 = train_and_evaluate_multivariate(
        scaled_cyl_data, using_columns_original, scaled_cyl_data.columns.tolist(),
        lag=LAG, n_splits=5
    )
    all_experiment_results['С цилиндрическими признаками'] = results_exp2

    print("\n--- Обзор результатов и важности признаков ---")
    for experiment_name, results_dict in all_experiment_results.items():
        print(f"\n======== {experiment_name} ========")
        for col, res in results_dict.items():
            plt.figure(figsize=(10, 4))
            plt.plot(res['y_true'], label='Истинные значения', linewidth=1.5)
            plt.plot(res['y_pred'], label='Предсказанные значения', linestyle='--')
            plt.title(f"Прогноз {col} ({experiment_name})")
            plt.xlabel("Шаг")
            plt.ylabel(col)
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()

            if 'feature_importances' in res and not res['feature_importances'].empty:
                print(f"\n--- Важность признаков для {col} (Топ-10) ---")
                print(res['feature_importances'].head(10))

                plt.figure(figsize=(10, 6))
                plt.barh(res['feature_importances']['Feature'].head(10),
                         res['feature_importances']['Importance'].head(10))
                plt.xlabel('Важность (F-score)')
                plt.ylabel('Признак')
                plt.title(f'Важность признаков для прогнозирования {col} ({experiment_name})')
                plt.gca().invert_yaxis()
                plt.tight_layout()
                plt.show()
            else:
                print(f"  Не удалось получить важность признаков для {col} в эксперименте '{experiment_name}'.")

    print("\n--- Визуализация цилиндрического представления всех сигналов ---")
    for col in using_columns_original:
        print(f"\nВизуализация цилиндра для: {col}")
        # Получаем период для текущего столбца
        T0 = periods[col]
        # Проверяем, достаточно ли данных для построения цилиндра с данным периодом
        if len(clean_df[col].values) > T0:
            plot_cylindrical_representation(clean_df[col].values, T0, col)  # Исправлено
        else:
            print(f"Недостаточно данных для построения цилиндра для {col} с периодом {T0}.")
